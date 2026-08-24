clc
clear all
clear global
close all;

addpath('scripts');

S.homepath = pwd;
S.utils = 'utils';
S.EEGdatafolder='filteredEEG';
S.datafolder='BehavioralDataAligned';
S.pathsymbol = '/';

% Which eeg data file to load
S.eegDataFile='filtedEEG';
S.behaveFile = '_behavioral_data';

% Exp 1
S.Subs = {'2', '3', '7', '10', '12', '13', '14', '15', '16', '19', '20', '21', '22', '23', '24', '25', '26', '29', '30', '31', '32', '33', '34', '36', '37', '38', '39', '40', '41', '42', '44', '45', '46', '47', '48', '49'};

% Split by median, good half
% S.Subs = {'2', '12', '14', '16', '19', '20', '21', '24', '26', '29', '33', '38', '39', '40', '42', '44', '47', '48'};

% Split by median, poor half
% S.Subs = {'3', '7', '10', '12', '13', '15', '22', '23', '25', '30', '31', '34', '36', '37', '41', '45', '46', '49'};

nsubs = size(S.Subs,2);


darkblue = [26, 128, 187]./255;
lightblue = [140, 197, 227]./255;
darkred = [160, 0, 0]./255;
lightred = [216, 166, 166]./255;

S.Baseline = 1; % 1=yes, 0=no baseline
S.BaselinBeg = -200;
S.BaselinEnd = 0;

startTime = -300; % ms
endTime = 798;    % ms
numPoints = 550;
times = linspace(startTime, endTime, numPoints);

S.PlotTimeBegin = -200;
S.PlotTimeEnd = 600;
S.PlotNegLim = -10;
S.PlotPosLim = +10;

S.meanBegin = 200;
S.meanEnd = 400;

%%% ------------------------------------------------------------------- %%%
% electrodes to ipison the data:

[LH, RH, flippedChannels] = ipsiconElectrodes();

S.IpsiconChannels = flippedChannels;

whichChannels = {'PO7/8', 'P7/8'};


% find electrodes numbers
for i = 1:numel(whichChannels); idxElec(i) = find(strcmp(S.IpsiconChannels, whichChannels(i)));end



%% ------------------------------------------------------------------- %%%

conds = [1 2]; % 1 = frequent, 2 = infrequent
condNames = {'Frequent', 'Infrequent'};
colors = lines(numel(S.Subs)); % one color per subject

%%  store different waves
train_ic_diff = cell(nsubs,2);
test_ic_diff  = cell(nsubs,2);

fprintf('Loading subjects and computing ERPs ...\n');
for SubNo = 1:size(S.Subs,2)
    disp(num2str(S.Subs{SubNo}));

    %----------------------------------------------------------------------
    % load eeg data
    eLoad = load([S.homepath S.pathsymbol S.EEGdatafolder S.pathsymbol 'Subject_' S.Subs{SubNo} S.eegDataFile '.mat']);
    e = eLoad.eegKept;
    
    % convert baseline time into indices    
    %----------------------------------------------------------------------
    % Load behavior and consolidate into vectors:
    d = load([S.homepath S.pathsymbol S.datafolder  S.pathsymbol 'Subject_' S.Subs{SubNo} S.behaveFile '.mat']);

    % vectorize
    ID = [d.data_table.ID];
    trialNumber = [d.data_table.TrialNumber];
    blockNumber = [d.data_table.BlockNumber];
    targID = [d.data_table.TargetID];
    targLoc = [d.data_table.TargetLocation];
    targCondition = [d.data_table.TargetCondition];
    acc = [d.data_table.Accuracy];
    rt = [d.data_table.RT];

    % --- drop slowest rt ---
    nTrials = numel(rt);
    [~, sortIdx] = sort(rt);  % indices sorted by RT
    cutoff = floor(4/5 * nTrials); % keep fastest proportions
    keepIdx = sortIdx(1:cutoff);   % indices of fastest trials
    
    % --- get trial indices ---
    trainIdx = keepIdx(blockNumber(keepIdx) <= 12);
    testIdx  = keepIdx(blockNumber(keepIdx) > 12);


    % Index EEG data using filtered RT
     %% --- Split train/test ---
    % trainEEG = eegKept(:,:,trainIdx);
    % testEEG  = eegKept(:,:,testIdx);
    % 
    % trainCond = targConditionKept(trainIdx);
    % testCond  = targConditionKept(testIdx);
    % trainLoc  = targLocKept(trainIdx);
    % testLoc   = targLocKept(testIdx);

    trainEEG = e(:,:,trainIdx);
    testEEG  = e(:,:,testIdx);

    trainCond = targCondition(trainIdx);
    testCond  = targCondition(testIdx);
    trainLoc  = targLoc(trainIdx);
    testLoc   = targLoc(testIdx);

     %% --- Baseline indices ---
    % begBaseline = find(e.times == S.BaselinBeg, 1);
    % endBaseline = find(e.times == S.BaselinEnd, 1);

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

    % Find the times to plot
    % plotBeg = find(e.times==(S.PlotTimeBegin),1);
    [~, plotBeg] = min(abs(times - S.PlotTimeBegin));
    % plotEnd = find(e.times==(S.PlotTimeEnd),1);
    [~, plotEnd] = min(abs(times - S.PlotTimeEnd));
    time_range = times(plotBeg:plotEnd);
    % begMean = find(e.times == S.meanBegin, 1);
    [~, begMean] = min(abs(times - S.meanBegin));
    % endMean = find(e.times == S.meanEnd, 1);
    [~, endMean] = min(abs(times - S.meanEnd));

    % Mean amplitudes
    icdifMean_FrequentTrain(SubNo) = squeeze(mean( ...
        mean(train_ic_diff{SubNo,1}(idxElec, begMean:endMean), 2), 1));

    icdifMean_InfrequentTrain(SubNo) = squeeze(mean( ...
        mean(train_ic_diff{SubNo,2}(idxElec, begMean:endMean), 2), 1));
    
    icdifMean_FrequentTest(SubNo) = squeeze(mean( ...
        mean(test_ic_diff{SubNo,1}(idxElec, begMean:endMean), 2), 1));
    
    icdifMean_InfrequentTest(SubNo) = squeeze(mean( ...
        mean(test_ic_diff{SubNo,2}(idxElec, begMean:endMean), 2), 1));

end

%% -----------------------------------------------------------------------
%% LATENCY CALCULATION — 50% fractional area under the curve
%  Measurement window: 175–600 ms, on the grand average difference wave
%  The N2pc is negative so we flip sign before computing area (same logic
%  as jackknife_latency.m), then find where cumulative area first exceeds
%  50% of total area within the window.
%% -----------------------------------------------------------------------

latencyWindow = [175, 600];  % ms

[~, latWinBeg] = min(abs(times - latencyWindow(1)));
[~, latWinEnd] = min(abs(times - latencyWindow(2)));

% Flip sign so N2pc is positive, zero out negatives (same as jackknife code)
function latMS = halfAreaLatency(wave, winBeg, winEnd, times)
    segment = wave(winBeg:winEnd);
    segment_pos = segment * -1;           % flip so N2pc is positive
    segment_pos(segment_pos < 0) = 0;    % zero out negatives
    totalArea = trapz(segment_pos);
    if totalArea <= 0
        latMS = NaN; return;
    end
    cumArea = cumtrapz(segment_pos);
    idx = find(cumArea >= totalArea / 2, 1, 'first');
    if isempty(idx)
        latMS = NaN; return;
    end
    latMS = times(winBeg + idx - 1);
end


% Plot ERP for each individual
%-------------------------------------------------------------------------
%% Train
figure; hold on;
set(gca, 'FontSize', 18, 'LineWidth', 3);

if nsubs < 1
    % --- Plot for one subject ---
    set(gca,'YDir','reverse');
    plot(time_range, mean(train_ic_diff{1}(:,idxElec,:),2), 'color',darkblue, 'LineWidth', 2); % frequent
    plot(time_range, mean(train_ic_diff{2}(:,idxElec,:),2), 'color',darkred, 'LineWidth', 2); % infrequent

else
    % --- Compute grand average across subjects ---
    fprintf('Grand averaging ...\n');

    grandTrain = zeros(2, numel(times));
    % grandTest  = zeros(2, numel(e.times));

    for c = 1:2   % 1 = frequent, 2 = infrequent
        for s = 1:nsubs
            % Average across channels of interest for this subject/condition
            subjTrain = mean(train_ic_diff{s, c}(idxElec,:), 1); % 1 × time
            % subjTest  = mean(test_ic_diff{s, c}(idxElec,:), 1);  % 1 × time

            trainData{s,c} = subjTrain;
            % testData{s,c}  = subjTest;
        end
    
        % Convert cell → matrix and average across subjects
        trainMat = cat(1, trainData{:,c});   % [nSubs x time]
        % testMat  = cat(1, testData{:,c});    % [nSubs x time]
    
        grandTrain(c,:) = mean(trainMat, 1); % [1 x time]
        % grandTest(c,:)  = mean(testMat, 1);  % [1 x time]
    end

    % Plot
    set(gca,'YDir','reverse');
    xline(0, 'k', 'LineWidth', 1, 'HandleVisibility', 'off');
    yline(0, 'k', 'LineWidth', 1, 'HandleVisibility', 'off');
    trainFreq = squeeze(grandTrain(1,:,:));
    trainInfreq = squeeze(grandTrain(2,:,:));
    lat_FreqTrain   = halfAreaLatency(trainFreq,   latWinBeg, latWinEnd, times);
    lat_InfreqTrain = halfAreaLatency(trainInfreq,  latWinBeg, latWinEnd, times);
    plot(time_range, trainInfreq(plotBeg:plotEnd), 'color', darkred, 'LineWidth', 2);
    plot(time_range, trainFreq(plotBeg:plotEnd), 'color', darkblue, 'LineWidth', 2);
    yticks([-2 -1.5 -1 -0.5 0 0.5]);
    ylim([-2 0.5]);    
    % --- Latency tick marks at the x-axis (y = 0.5, the top of the plot) ---
    tickY   = 0.5;   % place ticks at the top ylim edge
    tickH   = 0.08;  % height of tick in µV units
    
    if ~isnan(lat_FreqTrain)
        plot([lat_FreqTrain lat_FreqTrain], [tickY tickY - tickH], ...
            'color', darkblue, 'LineWidth', 2.5);
    end
    if ~isnan(lat_InfreqTrain)
        plot([lat_InfreqTrain lat_InfreqTrain], [tickY tickY - tickH], ...
            'color', darkred, 'LineWidth', 2.5);
    end
end
xlabel('Time (ms)'), ylabel(' µV'); hold on;
p1 = plot(time_range, trainInfreq(plotBeg:plotEnd), 'color', darkred,  'LineWidth', 2);
p2 = plot(time_range, trainFreq(plotBeg:plotEnd),   'color', darkblue, 'LineWidth', 2);
lg = legend([p2, p1], {'Frequent', 'Infrequent'});
lg.Position(1) = 0.65;
legend boxoff;

% Get the current figure handle
f = gcf;

% Export the figure as a 300 DPI PNG file
exportgraphics(f, 'N2pc_train_poor_80_200400.png', 'Resolution', 300);
% exportgraphics(f, 'N2pc_train.eps', 'ContentType', 'vector');

%% Test
figure; hold on;
set(gca, 'FontSize', 18, 'LineWidth', 3);


if nsubs < 1
    % --- Plot for one subject ---
    set(gca,'YDir','reverse');
    plot(time_range, mean(test_ic_diff{1}(:,idxElec,:),2), 'color',lightblue, 'LineWidth', 2); % frequent
    plot(time_range, mean(test_ic_diff{2}(:,idxElec,:),2), 'color',lightred, 'LineWidth', 2); % infrequent

else
    % --- Compute grand average across subjects ---
    fprintf('Grand averaging ...\n');

    % grandTrain = zeros(2, numel(e.times));
    grandTest  = zeros(2, numel(times));

    for c = 1:2   % 1 = frequent, 2 = infrequent
        for s = 1:nsubs
            % Average across channels of interest for this subject/condition
            % subjTrain = mean(train_ic_diff{s, c}(idxElec,:), 1); % 1 × time
            subjTest  = mean(test_ic_diff{s, c}(idxElec,:), 1);  % 1 × time

            % trainData{s,c} = subjTrain;
            testData{s,c}  = subjTest;
        end
    
        % Convert cell → matrix and average across subjects
        % trainMat = cat(1, trainData{:,c});   % [nSubs x time]
        testMat  = cat(1, testData{:,c});    % [nSubs x time]
    
        % grandTrain(c,:) = mean(trainMat, 1); % [1 x time]
        grandTest(c,:)  = mean(testMat, 1);  % [1 x time]
    end

    % Plot
    set(gca,'YDir','reverse');
    xline(0, 'k', 'LineWidth', 1, 'HandleVisibility', 'off');
    yline(0, 'k', 'LineWidth', 1, 'HandleVisibility', 'off');
    % trainFreq = squeeze(grandTrain(1,:,:));
    % trainInfreq = squeeze(grandTrain(2,:,:));
    % plot(times, trainFreq(plotBeg:plotEnd), 'color', darkblue, 'LineWidth', 2);
    % plot(times, trainInfreq(plotBeg:plotEnd), 'color', lightblue, 'LineWidth', 2);
    testFreq = squeeze(grandTest(1,:,:));
    testInfreq = squeeze(grandTest(2,:,:));
    lat_FreqTest    = halfAreaLatency(testFreq,    latWinBeg, latWinEnd, times);
    lat_InfreqTest  = halfAreaLatency(testInfreq,  latWinBeg, latWinEnd, times);
    yticks([-2,-1.5,-1,-0.5,0,0.5]);
    % latency ticks
    tickY = 0.5;
    tickH = 0.08;
    
    if ~isnan(lat_FreqTest)
        plot([lat_FreqTest lat_FreqTest], [tickY tickY - tickH], ...
            'color', lightblue, 'LineWidth', 2.5);
    end
    if ~isnan(lat_InfreqTest)
        plot([lat_InfreqTest lat_InfreqTest], [tickY tickY - tickH], ...
            'color', lightred, 'LineWidth', 2.5);
    end

    plot(time_range, testInfreq(plotBeg:plotEnd), 'color', lightred, 'LineWidth', 2);
    plot(time_range, testFreq(plotBeg:plotEnd), 'color', lightblue, 'LineWidth', 2);

end
xlabel('Time (ms)'), ylabel(' µV'); hold on;
p1 = plot(time_range, testInfreq(plotBeg:plotEnd), 'color', lightred,  'LineWidth', 2);
p2 = plot(time_range, testFreq(plotBeg:plotEnd),   'color', lightblue, 'LineWidth', 2);
lg = legend([p2, p1], {'Frequent', 'Infrequent'});
lg.Position(1) = 0.65;
legend boxoff;

% Get the current figure handle
f = gcf;

% Export the figure as a 300 DPI PNG file
exportgraphics(f, 'N2pc_test_poor_80_200400.png', 'Resolution', 300);
% exportgraphics(f, 'N2pc_test.eps', 'ContentType', 'vector');



figure(); clf;
set(gca, 'FontSize', 24, 'LineWidth', 3);hold on;
title('Mean amplitude', 'FontWeight', 'normal');
set(gca,'YDir','reverse'); box('off');
barSemWithin({icdifMean_FrequentTrain, icdifMean_InfrequentTrain, icdifMean_FrequentTest, icdifMean_InfrequentTest}, ...
    {'train freq', 'train infreq', 'test freq', 'test infreq'}, {darkblue, darkred, lightblue, lightred}); hold on;

% Get the current figure handle
f = gcf;

% Export the figure as a 300 DPI PNG file
exportgraphics(f, 'N2pc_amp_poor_80_200400.png', 'Resolution', 300);
% exportgraphics(f, 'N2pc_amp.eps', 'ContentType', 'vector');

fprintf('\n--- N2pc Latency (50%% fractional area, 175–600 ms) ---\n');
fprintf('Train Frequent:    %.1f ms\n', lat_FreqTrain);
fprintf('Train Infrequent:  %.1f ms\n', lat_InfreqTrain);
fprintf('Test  Frequent:    %.1f ms\n', lat_FreqTest);
fprintf('Test  Infrequent:  %.1f ms\n', lat_InfreqTest);

% ---------------------------------------------------------------------
% Calculate descriptive stats
% Means
mFreqTrain = mean(icdifMean_FrequentTrain);
mInfreqTrain = mean(icdifMean_InfrequentTrain);
mFreqTest = mean(icdifMean_FrequentTest);
mInfreqTest = mean(icdifMean_InfrequentTest);

% Standard deviations
sdFreqTrain = std(icdifMean_FrequentTrain);
sdInfreqTrain = std(icdifMean_InfrequentTrain);
sdFreqTest = std(icdifMean_FrequentTest);
sdInfreqTest = std(icdifMean_InfrequentTest);


% % ANOVA
data = [icdifMean_FrequentTrain(:), ...
        icdifMean_InfrequentTrain(:), ...
        icdifMean_FrequentTest(:), ...
        icdifMean_InfrequentTest(:)];
nSubs = length(S.Subs);

%% --- Reshape Data for ANOVAN (Wide to Long Format) ---

% Vector of all measurements (stacking the columns of 'data')
y = data(:); 
subject_group = repmat((1:nSubs)', 4, 1);
phase_group   = [repmat({'Train'}, 2*nSubs, 1); repmat({'Test'},  2*nSubs, 1)];
freq_group    = repmat([repmat({'Frequent'}, nSubs, 1); repmat({'Infrequent'}, nSubs, 1)], 2, 1);

[p, tbl, stats] = anovan(y, {subject_group, phase_group, freq_group}, ...
    'model', 'interaction', ...
    'random', 1, ...
    'varnames', {'Subject', 'Phase', 'Frequency'});
%% --- Effect sizes: generalized & partial eta^2 (from the anovan table) ---
% anovan may label interactions with '*' or ':', so normalize the separator.
src = regexprep(tbl(2:end,1), ':', '*');   % source names, header row removed
SS  = cell2mat(tbl(2:end,2));              % sums of squares, same row order
ssOf = @(term) SS(strcmp(src, term));

SS_subject   = ssOf('Subject');
SS_phase     = ssOf('Phase');
SS_freq      = ssOf('Frequency');
SS_subjPhase = ssOf('Subject*Phase');
SS_subjFreq  = ssOf('Subject*Frequency');
SS_phaseFreq = ssOf('Phase*Frequency');
SS_err       = ssOf('Error');

% Generalized eta^2: both factors are within-subject (manipulated), so the
% denominator pools ALL subject/error sources with the effect's own SS.
SS_errpool = SS_subject + SS_subjPhase + SS_subjFreq + SS_err;
etaG_phase     = SS_phase     / (SS_phase     + SS_errpool);
etaG_freq      = SS_freq      / (SS_freq      + SS_errpool);
etaG_phaseFreq = SS_phaseFreq / (SS_phaseFreq + SS_errpool);

% Partial eta^2: each effect over its own (correct) error term.
etap_phase     = SS_phase     / (SS_phase     + SS_subjPhase);
etap_freq      = SS_freq      / (SS_freq      + SS_subjFreq);
etap_phaseFreq = SS_phaseFreq / (SS_phaseFreq + SS_err);

fprintf('\n--- N2pc amplitude ANOVA effect sizes ---\n');
fprintf('Phase:       eta_G^2 = %.3f, eta_p^2 = %.3f\n', etaG_phase,     etap_phase);
fprintf('Frequency:   eta_G^2 = %.3f, eta_p^2 = %.3f\n', etaG_freq,      etap_freq);
fprintf('Phase*Freq:  eta_G^2 = %.3f, eta_p^2 = %.3f\n', etaG_phaseFreq, etap_phaseFreq);

% Post-hoc ttests
alpha_corrected = 0.025; % Bonferroni for 2 tests

[~, pTrain, ~, statsTrain] = ttest(icdifMean_FrequentTrain, icdifMean_InfrequentTrain);
[~, pTest,  ~, statsTest]  = ttest(icdifMean_FrequentTest,  icdifMean_InfrequentTest);

fprintf('Amplitude Train: t(%d) = %.3f, p = %.4f\n', statsTrain.df, statsTrain.tstat, pTrain);
fprintf('Amplitude Test:  t(%d) = %.3f, p = %.4f\n', statsTest.df,  statsTest.tstat,  pTest);


%% =====================================================================

%% ---- 1. Load channel locations ----
chanlocs = readlocs(fullfile(S.utils, '32channelsGreenwithDiode.loc'));
labels   = {chanlocs.labels};
nChan    = numel(chanlocs);

%% ---- 2. Build left<->right flip index ----
%  flipIdx(i) = row of the electrode that is the mirror image of channel i.
%  Midline / non-scalp channels map to themselves.
pairs = { ...
    'FP1','FP2'; 'F3','F4'; 'FC1','FC2'; 'FC5','FC6'; ...
    'T7','T8'; 'C3','C4'; 'CP1','CP2'; 'CP5','CP6'; ...
    'P7','P8'; 'P3','P4'; 'PO3','PO4'; 'PO7','PO8'; 'O1','O2' };

flipIdx = 1:nChan;                       % default: self
for p = 1:size(pairs,1)
    iL = find(strcmpi(labels, pairs{p,1}));
    iR = find(strcmpi(labels, pairs{p,2}));
    if ~isempty(iL) && ~isempty(iR)
        flipIdx(iL) = iR;
        flipIdx(iR) = iL;
    end
end
% Sanity check: flipping twice returns identity
assert(isequal(flipIdx(flipIdx), 1:nChan), 'flipIdx is not a valid involution');

%% ---- 3. Channels to exclude from the scalp plot ----
nonScalp = find(ismember(upper(labels), {'M1','HEOG','DIODE','M2','VEOG'}));

%% ---- 4. Time window 200-400 ms ----
[~, tBeg] = min(abs(times - 200));
[~, tEnd] = min(abs(times - 400));

%% ---- 5. Build contra-ipsi grand-average map per phase/condition ----
%  For each subject/condition:
%    contra map = mean over time of [ left-target map flipped + right-target map ] / 2
%                 ... wait: define per-electrode contra carefully (see notes).
%
%  Convention used here (standard N2pc collapsing):
%    - For RIGHT-field targets, contralateral = LEFT electrodes  -> use map as is
%    - For LEFT-field  targets, contralateral = RIGHT electrodes -> mirror the map
%    Averaging the right-target map with the FLIPPED left-target map yields,
%    at every electrode, the contralateral response. The ipsilateral map is
%    the same average WITHOUT flipping the left-target map (i.e. flip the
%    right-target map instead). contra - ipsi is the N2pc topography.
%
%  We then symmetrize for display so the focus appears on both sides.

condNames = {'Frequent','Infrequent'};
phaseNames = {'Train','Test'};

figure('Color','w','Name','N2pc contra-ipsi topography 200-400ms', ...
       'Position',[100 100 900 800]);
plotN = 1;

for phase = 1:2
    if phase==1
        Lsrc = train_left; Rsrc = train_right;
    else
        Lsrc = test_left;  Rsrc = test_right;
    end

    for c = 1:2
        ic_map = zeros(nChan, nsubs);
        for s = 1:nsubs
            Lmap = mean(Lsrc{s,c}(:, tBeg:tEnd), 2);   % chan x 1, left-field targets
            Rmap = mean(Rsrc{s,c}(:, tBeg:tEnd), 2);   % chan x 1, right-field targets

            % contralateral: right-target as-is + left-target mirrored
            contra = ( Rmap + Lmap(flipIdx) ) / 2;
            % ipsilateral: right-target mirrored + left-target as-is
            ipsi   = ( Rmap(flipIdx) + Lmap ) / 2;

            ic_map(:,s) = contra - ipsi;               % contra - ipsi
        end
        scalp = mean(ic_map, 2);                       % grand average

        % By construction scalp is antisymmetric (left = -right). To display
        % a conventional symmetric N2pc focus, plot the antisymmetric map
        % directly: negative focus contralateral, positive ipsilateral.
        scalpPlot = scalp;
        scalpPlot(nonScalp) = 0;                       % suppress non-scalp chans

        subplot(2,2,plotN); plotN = plotN+1;
        topoplot(scalpPlot, chanlocs, ...
                 'maplimits', [-1.5 1.5], ...          % adjust to your uV range
                 'electrodes','on', 'style','map', ...
                 'plotchans', setdiff(1:nChan, nonScalp));
        title(sprintf('%s - %s', phaseNames{phase}, condNames{c}));
        colorbar;
    end
end

exportgraphics(gcf, 'N2pc_topo_all.png', 'Resolution', 300);



