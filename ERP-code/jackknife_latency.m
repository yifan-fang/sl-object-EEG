close all
clc
clear all
clear global

%% =======================================================================
%  N2pc half-area latency, jackknifed, with a 2 (Frequency) x 2 (Phase)
%  within-subjects ANOVA across a grid of measurement windows,
%  computed AFTER dropping the slowest 1/5 of trials.
%
%  Combines:
%    n2pcbyRT.m          -- trial-level RT filtering and ipsi/contra binning
%    jackknife_latency.m -- jackknife half-area latency + 2x2 ANOVA
%
%  IMPORTANT: this uses trial-level EEG (eegKept, chan x time x trials),
%  NOT the pre-averaged ERP bins (e.bindata, chan x time x 12 bins).
%  Bin-averaged data cannot be filtered by RT because individual trials
%  no longer exist in it.
%
%  Condition order in diffWaves / jackAreaLatency:
%     1 = Frequent  / Train      3 = Frequent  / Test
%     2 = Infrequent/ Train      4 = Infrequent/ Test
%
%  Stuff marked %CHANGETHIS may need editing.
%  =======================================================================

%% Paths and files
S.homepath      = pwd;
S.utils         = 'utils';
S.EEGdatafolder = 'filteredEEG';            % trial-level EEG lives here
S.datafolder    = 'BehavioralDataAligned';
S.pathsymbol    = '/';

addpath('scripts');
addpath('utils');

S.eegDataFile = 'filtedEEG';                % -> variable eegKept
S.behaveFile  = '_behavioral_data';         % -> variable data_table

% Exp 1
S.Subs = {'2', '3', '7', '10', '12', '13', '14', '15', '16', '19', '20', '21', '22', '23', '24', '25', '26', '29', '30', '31', '32', '33', '34', '36', '37', '38', '39', '40', '41', '42', '44', '45', '46', '47', '48', '49'};

% Split by median, good half
% S.Subs = {'2', '12', '14', '16', '19', '20', '21', '24', '26', '29', '33', '38', '39', '40', '42', '44', '47', '48'};

% Split by median, poor half
% S.Subs = {'3', '7', '10', '12', '13', '15', '22', '23', '25', '30', '31', '34', '36', '37', '41', '45', '46', '49'};

nSubsToUse = numel(S.Subs);

%% RT filtering options  %CHANGETHIS
S.rtKeepFraction = 4/5;      % keep the fastest 4/5, i.e. drop slowest 1/5

% 'cell'    = drop the slowest fraction WITHIN each phase x frequency cell.
%             Keeps trial counts balanced across the four cells.
% 'session' = drop the slowest fraction across the whole session (the
%             literal "drop the slowest 1/5 of trials"). Slower conditions
%             lose disproportionately more trials.
S.rtDropScope = 'cell';

%% Timing
% eegKept epochs: -300 to 798 ms, 550 points -> 2 ms per sample (500 Hz)
startTime = -300;
endTime   = 798;
numPoints = 550;
times     = linspace(startTime, endTime, numPoints);
msPerTP   = (endTime - startTime) / (numPoints - 1);   % 2 ms

S.BaselinBeg = -200;
S.BaselinEnd = 0;

% Nearest-sample index. Do NOT use find(times==x,1): linspace output is
% floating point and exact equality can silently fail.
tIdx = @(ms) find(abs(times - ms) == min(abs(times - ms)), 1);

begBaseline = tIdx(S.BaselinBeg);
endBaseline = tIdx(S.BaselinEnd);

%% Electrodes
[LH, RH, flippedChannels] = ipsiconElectrodes();
S.IpsiconChannels = flippedChannels;

whichChannels = {'PO7/8', 'P7/8'};
for i = 1:numel(whichChannels)
    idxElec(i) = find(strcmp(S.IpsiconChannels, whichChannels(i)));
end

conds      = [1 2];                          % 1 = frequent, 2 = infrequent
condLabels = {'Frequent Train','Infrequent Train','Frequent Test','Infrequent Test'};

%% =======================================================================
%  STAGE 1: load trials, drop slowest RTs, build contra-ipsi difference waves
%  =======================================================================

diffWaves   = nan(numel(idxElec), numPoints, 4, nSubsToUse);  % chan x time x cond x sub
trialCounts = nan(nSubsToUse, 4);
rtCellMeans = nan(nSubsToUse, 4);
propDropped = nan(nSubsToUse, 1);

fprintf('Loading subjects, dropping slowest %.0f%% of trials (scope: %s) ...\n', ...
    (1-S.rtKeepFraction)*100, S.rtDropScope);

for SubNo = 1:nSubsToUse
    fprintf('Subject %s\n', S.Subs{SubNo});

    % ---- trial-level EEG: chan x time x trials ----
    eLoad = load([S.homepath S.pathsymbol S.EEGdatafolder S.pathsymbol ...
                  'Subject_' S.Subs{SubNo} S.eegDataFile '.mat']);
    e = eLoad.eegKept;

    % ---- aligned behavior ----
    d = load([S.homepath S.pathsymbol S.datafolder S.pathsymbol ...
              'Subject_' S.Subs{SubNo} S.behaveFile '.mat']);

    blockNumber   = [d.data_table.BlockNumber];
    targLoc       = [d.data_table.TargetLocation];
    targCondition = [d.data_table.TargetCondition];
    rt            = [d.data_table.RT];

    nTrials = numel(rt);
    [~, sortIdx] = sort(rt);  % indices sorted by RT
    cutoff = floor(4/5 * nTrials); % keep fastest proportions
    keepIdx = sortIdx(1:cutoff);   % indices of fastest trials
    trainIdx = keepIdx(blockNumber(keepIdx) <= 12);
    testIdx  = keepIdx(blockNumber(keepIdx) > 12);

    trainEEG = e(:,:,trainIdx);
    testEEG  = e(:,:,testIdx);

    trainCond = targCondition(trainIdx);
    testCond  = targCondition(testIdx);
    trainLoc  = targLoc(trainIdx);
    testLoc   = targLoc(testIdx);

    [~, begBaseline] = min(abs(times - S.BaselinBeg));
    [~, endBaseline] = min(abs(times - S.BaselinEnd));

    for c = 1:2
        %% --- Train ERP ---
        condTrainIdx = find(trainCond == conds(c));
        thisTrainEEG = trainEEG(:,:,condTrainIdx);   % chan x time x nTrials
    
        % baseline correction trial-by-trial
        for tr = 1:size(thisTrainEEG,3)
            baselineMean = mean(thisTrainEEG(:,begBaseline:endBaseline,tr), 2);
            thisTrainEEG(:,:,tr) = thisTrainEEG(:,:,tr) - baselineMean;
        end

        %% --- Test ERP ---
        condTestIdx = find(testCond == conds(c));
        thisTestEEG = testEEG(:,:,condTestIdx);
        
        for tr = 1:size(thisTestEEG,3)
            baselineMean = mean(thisTestEEG(:,begBaseline:endBaseline,tr), 2);
            thisTestEEG(:,:,tr) = thisTestEEG(:,:,tr) - baselineMean;
        end


        %% --- Ipsilateral / Contralateral ERPs ---
        % Train
        trainEEG_corrected = thisTrainEEG;
        leftTrials  = trainLoc(condTrainIdx) < 3;
        rightTrials = trainLoc(condTrainIdx) > 2;
        train_ipsi   = (mean(trainEEG_corrected(LH,:,leftTrials),3) + mean(trainEEG_corrected(RH,:,rightTrials),3)) / 2;
        train_contra = (mean(trainEEG_corrected(LH,:,rightTrials),3) + mean(trainEEG_corrected(RH,:,leftTrials),3)) / 2;
        train_ic_diff{SubNo, c} = train_contra - train_ipsi;

        train_left{SubNo,c}  = mean(trainEEG_corrected(:,:,leftTrials),  3); % chan x time
        train_right{SubNo,c} = mean(trainEEG_corrected(:,:,rightTrials), 3);
    
        % Test
        testEEG_corrected = thisTestEEG;
        leftTrials  = testLoc(condTestIdx) < 3;
        rightTrials = testLoc(condTestIdx) > 2;
        test_ipsi   = (mean(testEEG_corrected(LH,:,leftTrials),3) + mean(testEEG_corrected(RH,:,rightTrials),3)) / 2;
        test_contra = (mean(testEEG_corrected(LH,:,rightTrials),3) + mean(testEEG_corrected(RH,:,leftTrials),3)) / 2;
        test_ic_diff{SubNo, c} = test_contra - test_ipsi;

        test_left{SubNo,c}   = mean(testEEG_corrected(:,:,leftTrials),   3);
        test_right{SubNo,c}  = mean(testEEG_corrected(:,:,rightTrials),  3);
    end
end

nSubsToUse = size(train_ic_diff, 1);
numPoints  = size(train_ic_diff{1,1}, 2);

diffWaves = nan(numel(idxElec), numPoints, 4, nSubsToUse);

for SubNo = 1:nSubsToUse
    diffWaves(:,:,1,SubNo) = train_ic_diff{SubNo,1}(idxElec,:);   % Frequent Train
    diffWaves(:,:,2,SubNo) = train_ic_diff{SubNo,2}(idxElec,:);   % Infrequent Train
    diffWaves(:,:,3,SubNo) = test_ic_diff{SubNo,1}(idxElec,:);    % Frequent Test
    diffWaves(:,:,4,SubNo) = test_ic_diff{SubNo,2}(idxElec,:);    % Infrequent Test
end


%% =======================================================================
%  STAGE 2: jackknife half-area latency
%  =======================================================================

alpha = 0.05;
df    = nSubsToUse - 1;

fCritEffect = finv(1 - alpha, 1, nSubsToUse-1);   % ~4.12 at n=36
tCritEffect = tinv(1 - alpha/2,   nSubsToUse-1);  % ~2.03 at n=36, two-tailed

jackCorrection = (nSubsToUse-1)^2;   % F inflation from jackknifing

EEGtimes = times;

% Leave-one-out grand averages; last slice is the real grand average
gavgDifWavesJ = zeros(size(diffWaves,1), size(diffWaves,2), size(diffWaves,3), nSubsToUse+1);
for s = 1:nSubsToUse
    jackLogic = true(1, nSubsToUse);
    jackLogic(s) = false;
    gavgDifWavesJ(:,:,:,s) = mean(diffWaves(:,:,:,jackLogic), 4);
end
gavgDifWavesJ(:,:,:,nSubsToUse+1) = mean(diffWaves, 4);

% Average across channels -> time x conditions x (jackERPs + gavg)
colDifWavesJ = squeeze(mean(gavgDifWavesJ, 1));

% Flip so the N2pc is positive and clip negatives. Clipping is required by
% the half-area method: mixed-sign excursions cancel and the cumulative
% area can cross the halfway mark more than once, or never.
colDifWavesJack = colDifWavesJ * -1;
colDifWavesJack(colDifWavesJack < 0) = 0;

% Measurement window grid (ms)  %CHANGETHIS
windowStarts = 174:10:500;
windowEnds   = 278:10:598;

minimumMeasurementWindowTP = round(100 / msPerTP);   % 100 ms

[wX, wY]   = meshgrid(windowStarts, windowEnds);
[wiX, wiY] = meshgrid(1:length(windowStarts), 1:length(windowEnds));
windows    = [wX(:) wY(:)];

nStart = length(windowStarts);
nEnd   = length(windowEnds);

% slice 1 = Frequency main, 2 = Phase main, 3 = Frequency x Phase
tSpace       = nan(nStart, nEnd, 3);   % signed jackknife t (already corrected)
fSpace       = nan(nStart, nEnd, 3);   % jackknife-corrected F
pSpace       = nan(nStart, nEnd, 3);   % p from corrected F
etaGSpace    = nan(nStart, nEnd, 3);   % generalized eta^2
tSpaceSimple = nan(nStart, nEnd, 2);   % 1 = train, 2 = test simple effects
latencySpace = nan(nStart, nEnd, 4);   % measured latency (ms) per condition

for win = 1:size(windows,1)

    windowBeg = windows(win,1); windowEnd = windows(win,2);
    windowBegT = tIdx(windowBeg);
    windowEndT = tIdx(windowEnd);

    adjust = windowBegT - 1;

    if (windowEndT - windowBegT) < minimumMeasurementWindowTP
        continue    % surfaces are preallocated NaN
    end

    clear areas subAreas jackAreaLatency halfAreaTests halfAreaLogic
    clear jackAnovaDifs jAnovaMeans jAnovaSEs jTsAnova
    clear jackLatencyDifs jMeans jSEs jTs
    clear tblA pA statsA srcNames

    % total area (half-area denominator)
    for subERPJack = 1:size(colDifWavesJack,3)
        for cond = 1:size(colDifWavesJack,2)
            areas(cond,subERPJack) = trapz(colDifWavesJack(windowBegT:windowEndT,cond,subERPJack));
        end
    end

    % cumulative sub-areas: conditions x subERPs x timepoints
    for subERPJack = 1:size(colDifWavesJack,3)
        for cond = 1:size(colDifWavesJack,2)
            subAreas(cond,subERPJack,:) = cumtrapz(colDifWavesJack(windowBegT:windowEndT,cond,subERPJack));
        end
    end

    halfAreaTests = subAreas - areas/2;   %CHANGETHIS for a different fraction
    halfAreaLogic = halfAreaTests > 0;

    for subERPJack = 1:size(colDifWavesJack,3)
        for cond = 1:size(colDifWavesJack,2)
            try
                jackAreaLatency(cond,subERPJack) = find(halfAreaLogic(cond,subERPJack,:),1,'first');
            catch
                jackAreaLatency(cond,subERPJack) = NaN;
            end
        end
    end

    % A missing cell unbalances the design and invalidates the (n-1)^2
    % correction, so skip the window rather than fitting on it.
    if any(isnan(jackAreaLatency(:,1:nSubsToUse)), 'all')
        continue
    end

    measuredLatencyTP = jackAreaLatency(:,nSubsToUse+1);
    measuredLatencyMS = EEGtimes(measuredLatencyTP + adjust);

    %% --- three 1-df signed jackknife contrasts ---
    JAL = jackAreaLatency;

    % Frequency: infrequent - frequent, collapsed over phase
    jackAnovaDifs(1,:) = (JAL(2,:)+JAL(4,:))/2 - (JAL(1,:)+JAL(3,:))/2;
    % Phase: test - train, collapsed over frequency
    jackAnovaDifs(2,:) = (JAL(3,:)+JAL(4,:))/2 - (JAL(1,:)+JAL(2,:))/2;
    % Interaction: (infreq-freq) in test minus (infreq-freq) in train
    jackAnovaDifs(3,:) = (JAL(4,:)-JAL(3,:)) - (JAL(2,:)-JAL(1,:));

    jAnovaMeans = mean(jackAnovaDifs(:,1:nSubsToUse),2);
    jAnovaSEs   = sqrt( ((nSubsToUse-1)/nSubsToUse) * ...
                        sum((jackAnovaDifs(:,1:nSubsToUse)-jAnovaMeans).^2, 2) );
    jTsAnova    = jackAnovaDifs(:,nSubsToUse+1) ./ jAnovaSEs;   % already corrected

    % Simple effects within each phase
    jackLatencyDifs(1,:) = JAL(2,:) - JAL(1,:);
    jackLatencyDifs(2,:) = JAL(4,:) - JAL(3,:);
    jMeans = mean(jackLatencyDifs(:,1:nSubsToUse),2);
    jSEs   = sqrt( ((nSubsToUse-1)/nSubsToUse) * ...
                   sum((jackLatencyDifs(:,1:nSubsToUse)-jMeans).^2, 2) );
    jTs    = jackLatencyDifs(:,nSubsToUse+1) ./ jSEs;
    jMeanLatency  = mean(jackAreaLatency(:,1:nSubsToUse), 2);
    jSEsLatencyTP = sqrt( ((nSubsToUse-1)/nSubsToUse) * ...
                      sum((jackAreaLatency(:,1:nSubsToUse) - jMeanLatency).^2, 2) );
    jSEsLatencyMS = jSEsLatencyTP * msPerTP;

    %% --- anovan 2x2, subject random ---
    subNumber = (1:nSubsToUse)';

    Latency4 = [JAL(1,1:nSubsToUse)'; JAL(2,1:nSubsToUse)'; ...
                JAL(3,1:nSubsToUse)'; JAL(4,1:nSubsToUse)'];

    Frequency = [ones(nSubsToUse,1); ones(nSubsToUse,1)*2; ...
                 ones(nSubsToUse,1); ones(nSubsToUse,1)*2];
    Phase     = [ones(nSubsToUse,1); ones(nSubsToUse,1); ...
                 ones(nSubsToUse,1)*2; ones(nSubsToUse,1)*2];
    SN4       = repmat(subNumber,4,1);

    % Freq, Phase, Subj, Freq*Phase, Freq*Subj, Phase*Subj
    % Three-way deliberately excluded so it becomes the residual.
    anovaTerms = [1 0 0; 0 1 0; 0 0 1; 1 1 0; 1 0 1; 0 1 1];

    [pA, tblA, statsA] = anovan(Latency4, {Frequency Phase SN4}, ...
        'model', anovaTerms, 'random', 3, ...
        'varnames', {'Frequency','Phase','subject'}, 'display', 'off');

    % Locate rows/cols by name: 'random' shifts the layout and MATLAB
    % writes interactions with ':' not '*'
    srcNames = tblA(:,1);
    fCol  = find(strcmp(tblA(1,:), 'F'),       1);
    dfCol = find(strcmp(tblA(1,:), 'd.f.'),    1);
    ssCol = find(strcmp(tblA(1,:), 'Sum Sq.'), 1);

    rowFreq  = find(strcmp(srcNames, 'Frequency'), 1);
    rowPhase = find(strcmp(srcNames, 'Phase'),     1);
    rowInt   = find(contains(srcNames,'Frequency') & contains(srcNames,'Phase'), 1);
    rowSubj  = find(strcmp(srcNames, 'subject'),   1);

    rowErrFreq  = find(strcmp(srcNames, 'Frequency:subject'), 1);
    rowErrPhase = find(strcmp(srcNames, 'Phase:subject'),     1);
    rowErrInt   = find(strcmp(srcNames, 'Error'),             1);

    if any(cellfun(@isempty, {fCol,dfCol,ssCol,rowFreq,rowPhase,rowInt, ...
                              rowSubj,rowErrFreq,rowErrPhase,rowErrInt}))
        error('ANOVA table layout unexpected. Sources: %s', ...
              strjoin(string(srcNames(2:end))', ', '));
    end

    
    fEffect = [tblA{rowFreq,fCol}; tblA{rowPhase,fCol}; tblA{rowInt,fCol}] / jackCorrection;
    dfNum   = [tblA{rowFreq,dfCol}; tblA{rowPhase,dfCol}; tblA{rowInt,dfCol}];
    dfDen   = [tblA{rowErrFreq,dfCol}; tblA{rowErrPhase,dfCol}; tblA{rowErrInt,dfCol}];
    pEffect = 1 - fcdf(fEffect, dfNum, dfDen);

    %% --- effect sizes ---
    % Effect SS are unaffected by jackknifing (marginal means equal the real
    % grand average). All subject-involving SS are shrunk by (n-1)^2 and
    % must be scaled back up.
    ssEff = [tblA{rowFreq,ssCol}; tblA{rowPhase,ssCol}; tblA{rowInt,ssCol}];
    ssSubjPool = ( tblA{rowSubj,ssCol}     + tblA{rowErrFreq,ssCol} + ...
                   tblA{rowErrPhase,ssCol} + tblA{rowErrInt,ssCol} ) * jackCorrection;

    etaG = ssEff ./ (ssEff + ssSubjPool);

    % partial eta^2 two ways, as a check that ssCol is the right column
    etaP        = fEffect .* dfNum ./ (fEffect .* dfNum + dfDen);
    ssErrScaled = [tblA{rowErrFreq,ssCol}; tblA{rowErrPhase,ssCol}; ...
                   tblA{rowErrInt,ssCol}] * jackCorrection;
    etaP_fromSS = ssEff ./ (ssEff + ssErrScaled);

    %% --- save ---
    for eff = 1:3
        tSpace(wiX(win),wiY(win),eff)    = jTsAnova(eff);
        fSpace(wiX(win),wiY(win),eff)    = fEffect(eff);
        pSpace(wiX(win),wiY(win),eff)    = pEffect(eff);
        etaGSpace(wiX(win),wiY(win),eff) = etaG(eff);
    end
    tSpaceSimple(wiX(win),wiY(win),1) = jTs(1);
    tSpaceSimple(wiX(win),wiY(win),2) = jTs(2);
    latencySpace(wiX(win),wiY(win),:) = measuredLatencyMS;

    % Validate both routes and the SS lookup, once
    if ~exist('routeChecked','var')
        fprintf('\n--- Route cross-check at window %d-%d ms ---\n', windowBeg, windowEnd);
        fprintf('  F from anovan (corrected): %8.4f %8.4f %8.4f\n', fEffect);
        fprintf('  t^2 from contrasts:        %8.4f %8.4f %8.4f\n', jTsAnova.^2);
        fprintf('  denominator df:            %8d %8d %8d\n', dfDen);
        fprintf('  eta2p from F:              %8.4f %8.4f %8.4f\n', etaP);
        fprintf('  eta2p from SS:             %8.4f %8.4f %8.4f\n', etaP_fromSS);
        fprintf('  eta2G:                     %8.4f %8.4f %8.4f\n', etaG);
        routeChecked = true;
    end

end % windows

tSpace(isinf(tSpace)) = nan;
fSpace(isinf(fSpace)) = nan;
tSpaceSimple(isinf(tSpaceSimple)) = nan;

%% =======================================================================
%  STAGE 3: plotting
%  =======================================================================

effectNames = {'Frequency (main)','Phase (main)','Frequency \times Phase'};
simpleNames = {'Train: infrequent - frequent','Test: infrequent - frequent'};
gray_color  = [0.7 0.7 0.7];

nc = 256; hc = nc/2;
cmapDiv = [linspace(0.10,1,hc)' linspace(0.25,1,hc)' linspace(0.65,1,hc)'; ...
           ones(hc,1) linspace(1,0.25,hc)' linspace(1,0.10,hc)'];

startTimes = windowStarts;
endTimes   = windowEnds;
[EndGrid, StartGrid] = meshgrid(endTimes, startTimes);
x = EndGrid(:); y = StartGrid(:); z = zeros(size(x));

titleSuffix = sprintf('fastest %.0f%% of trials, %s-wise', ...
    S.rtKeepFraction*100, S.rtDropScope);

%% F surfaces
figure(); set(gcf,'Units','Normalized','OuterPosition',[0 0.25 1 0.6]);
tiledlayout(1,3);
sgtitle(sprintf('N2pc latency: jackknife-corrected F (%s)', titleSuffix))

fmax = max(fSpace(:),[],'omitnan');
if isempty(fmax) || ~isfinite(fmax) || fmax <= fCritEffect
    fmax = fCritEffect + 1;
end

for eff = 1:3
    c = fSpace(:,:,eff); c = c(:);
    sig = c > fCritEffect;

    nexttile; hold on;
    scatter3(x(~sig), y(~sig), z(~sig), 100, gray_color, 'filled');
    scatter3(x(sig),  y(sig),  z(sig),  100, c(sig),     'filled');
    colormap(gca, autumn(256)); colorbar; clim([fCritEffect fmax]);

    title(sprintf('%s  (%d sig)', effectNames{eff}, sum(sig)));
    xlabel('End Time (ms)'); ylabel('Start Time (ms)');
    xticks(endTimes(1):20:endTimes(end)); yticks(startTimes(1):20:startTimes(end));
    view(2); axis tight; set(gca,'YDir','reverse');
end

%% t surfaces (signed)
figure(); set(gcf,'Units','Normalized','OuterPosition',[0 0.25 1 0.6]);
tiledlayout(1,3);
sgtitle(sprintf('N2pc latency: jackknife t, signed (%s)', titleSuffix))

tabs = max(abs(tSpace(:)),[],'omitnan');
if isempty(tabs) || ~isfinite(tabs) || tabs <= 0, tabs = tCritEffect + 1; end

for eff = 1:3
    c = tSpace(:,:,eff); c = c(:);
    sig = abs(c) > tCritEffect;

    nexttile; hold on;
    scatter3(x(~sig), y(~sig), z(~sig), 100, gray_color, 'filled');
    scatter3(x(sig),  y(sig),  z(sig),  100, c(sig),     'filled');
    colormap(gca, cmapDiv); colorbar; clim([-tabs tabs]);

    title(sprintf('%s  (%d sig)', effectNames{eff}, sum(sig)));
    xlabel('End Time (ms)'); ylabel('Start Time (ms)');
    xticks(endTimes(1):20:endTimes(end)); yticks(startTimes(1):20:startTimes(end));
    view(2); axis tight; set(gca,'YDir','reverse');
end

%% Simple effects (delete this block if not needed)
figure(); set(gcf,'Units','Normalized','OuterPosition',[0 0.25 0.7 0.6]);
tiledlayout(1,2);
sgtitle(sprintf('Simple effects: jackknife t, signed (%s)', titleSuffix))

tabsS = max(abs(tSpaceSimple(:)),[],'omitnan');
if isempty(tabsS) || ~isfinite(tabsS) || tabsS <= 0, tabsS = tCritEffect + 1; end

for ts = 1:2
    c = tSpaceSimple(:,:,ts); c = c(:);
    sig = abs(c) > tCritEffect;

    nexttile; hold on;
    scatter3(x(~sig), y(~sig), z(~sig), 100, gray_color, 'filled');
    scatter3(x(sig),  y(sig),  z(sig),  100, c(sig),     'filled');
    colormap(gca, cmapDiv); colorbar; clim([-tabsS tabsS]);

    title(sprintf('%s  (%d sig)', simpleNames{ts}, sum(sig)));
    xlabel('End Time (ms)'); ylabel('Start Time (ms)');
    xticks(endTimes(1):20:endTimes(end)); yticks(startTimes(1):20:startTimes(end));
    view(2); axis tight; set(gca,'YDir','reverse');
end

%% Grand average difference waves, for a sanity check on the components
figure(); set(gcf,'Units','Normalized','OuterPosition',[0.1 0.1 0.6 0.7]);
plotBeg = tIdx(-200); plotEnd = tIdx(600);
colorsCond = {[26 128 187]/255, [160 0 0]/255, [140 197 227]/255, [216 166 166]/255};
hold on; set(gca,'FontSize',14,'LineWidth',2,'YDir','reverse'); box off;
for cond = 1:4
    wv = colDifWavesJ(:,cond,nSubsToUse+1);
    plot(times(plotBeg:plotEnd), wv(plotBeg:plotEnd), ...
        'Color', colorsCond{cond}, 'LineWidth', 2);
end
yline(0,'k-'); xline(0,'k-');
legend(condLabels, 'Location','southeast'); legend boxoff;
xlabel('Time (ms)'); ylabel('Contra - Ipsi (\muV)');
title(sprintf('N2pc difference waves (%s)', titleSuffix), 'FontWeight','normal');

%% =======================================================================
%  Report the pre-registered window
%  =======================================================================
reportStart = 174; reportEnd = 598;   %CHANGETHIS
ri = find(windowStarts == reportStart, 1);
ci = find(windowEnds   == reportEnd,   1);

if ~isempty(ri) && ~isempty(ci) && ~isnan(fSpace(ri,ci,1))
    fprintf('\n=== Window %d-%d ms, fastest %.0f%% of trials (%s-wise) ===\n', ...
        reportStart, reportEnd, S.rtKeepFraction*100, S.rtDropScope);

    for cond = 1:4
        fprintf('  %-18s latency = %6.1f ms\n', condLabels{cond}, latencySpace(ri,ci,cond));
    end
    for eff = 1:3
        fprintf('  %-22s F(1,%d) = %6.3f, p = %.4f, eta2G = %.4f, t = %+6.3f\n', ...
            effectNames{eff}, nSubsToUse-1, fSpace(ri,ci,eff), ...
            pSpace(ri,ci,eff), etaGSpace(ri,ci,eff), tSpace(ri,ci,eff));
    end
    for ts = 1:2
        fprintf('  %-30s t(%d) = %+6.3f\n', simpleNames{ts}, ...
            nSubsToUse-1, tSpaceSimple(ri,ci,ts));
    end
else
    fprintf('\nWindow %d-%d ms was not fitted (too short, or NaN latency).\n', ...
        reportStart, reportEnd);
end