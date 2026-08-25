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
% S.erpDataFile = 'ERPfilt';
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
black = [0, 0, 0]./255;

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

% peak at 310ms after collapse
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
targetIndex_ind = nan(nsubs,1); 

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

    %% --- Frequent target category ---
    subjID = ID(1);
    targetIndex_ind(SubNo) = mod(subjID - 1, 4) + 1;

     %% --- Baseline indices ---
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

        % --- per-subject mean RT, matched to kept trials + phase + condition ---
        trIdxC = trainIdx(targCondition(trainIdx) == conds(c));   % train, this cond
        teIdxC = testIdx (targCondition(testIdx)  == conds(c));   % test,  this cond
        rt_train(SubNo,c) = mean(rt(trIdxC), 'omitnan');
        rt_test (SubNo,c) = mean(rt(teIdxC), 'omitnan');

         % --- stash trial-level data for split-half reliability ---
        % Train
        SH.train{SubNo,c}.eeg  = trainEEG_corrected;          % chan x time x nTrials 
        SH.train{SubNo,c}.left = trainLoc(condTrainIdx) < 3;  % left-target trials
        SH.train{SubNo,c}.right= trainLoc(condTrainIdx) > 2;
        % Test
        SH.test{SubNo,c}.eeg   = testEEG_corrected;
        SH.test{SubNo,c}.left  = testLoc(condTestIdx) < 3;
        SH.test{SubNo,c}.right = testLoc(condTestIdx) > 2;

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

%% --- Individual-subject N2pc latencies (50% fractional area) ---
lat_FreqTrain_ind   = nan(nsubs,1);
lat_InfreqTrain_ind = nan(nsubs,1);
lat_FreqTest_ind    = nan(nsubs,1);
lat_InfreqTest_ind  = nan(nsubs,1);

for s = 1:nsubs
    % collapse across the electrodes of interest -> 1 x time
    wTrainFreq   = mean(train_ic_diff{s,1}(idxElec,:), 1);
    wTrainInfreq = mean(train_ic_diff{s,2}(idxElec,:), 1);
    wTestFreq    = mean(test_ic_diff{s,1}(idxElec,:),  1);
    wTestInfreq  = mean(test_ic_diff{s,2}(idxElec,:),  1);

    lat_FreqTrain_ind(s)   = halfAreaLatency(wTrainFreq,   latWinBeg, latWinEnd, times);
    lat_InfreqTrain_ind(s) = halfAreaLatency(wTrainInfreq, latWinBeg, latWinEnd, times);
    lat_FreqTest_ind(s)    = halfAreaLatency(wTestFreq,    latWinBeg, latWinEnd, times);
    lat_InfreqTest_ind(s)  = halfAreaLatency(wTestInfreq,  latWinBeg, latWinEnd, times);
end

%% --- Export individual N2pc latencies to CSV (long format) ---
phaseCol = {};
freqCol  = {};
subjCol  = {};
latCol   = [];

condDef = {
    'Train', 'Frequent',   lat_FreqTrain_ind
    'Train', 'Infrequent', lat_InfreqTrain_ind
    'Test',  'Frequent',   lat_FreqTest_ind
    'Test',  'Infrequent', lat_InfreqTest_ind
};

for i = 1:size(condDef,1)
    lat = condDef{i,3};
    for s = 1:nsubs
        subjCol{end+1,1}  = S.Subs{s};       % subject ID as given (string)
        phaseCol{end+1,1} = condDef{i,1};
        freqCol{end+1,1}  = condDef{i,2};
        latCol(end+1,1)   = lat(s);          % NaN preserved for unmeasurable subjects
    end
end

T = table(subjCol, phaseCol, freqCol, latCol, ...
    'VariableNames', {'subject','phase','frequency','latency'});

writetable(T, 'individual_N2pc_latencies.csv');

% report (omitnan in case some subjects have no measurable area)
fprintf('\n--- Individual N2pc latencies (mean ± SD across subjects) ---\n');
fprintf('Train Frequent:   %.1f ± %.1f ms  (n=%d valid)\n', ...
    mean(lat_FreqTrain_ind,'omitnan'),   std(lat_FreqTrain_ind,'omitnan'),   sum(~isnan(lat_FreqTrain_ind)));
fprintf('Train Infrequent: %.1f ± %.1f ms  (n=%d valid)\n', ...
    mean(lat_InfreqTrain_ind,'omitnan'), std(lat_InfreqTrain_ind,'omitnan'), sum(~isnan(lat_InfreqTrain_ind)));
fprintf('Test  Frequent:   %.1f ± %.1f ms  (n=%d valid)\n', ...
    mean(lat_FreqTest_ind,'omitnan'),    std(lat_FreqTest_ind,'omitnan'),    sum(~isnan(lat_FreqTest_ind)));
fprintf('Test  Infrequent: %.1f ± %.1f ms  (n=%d valid)\n', ...
    mean(lat_InfreqTest_ind,'omitnan'),  std(lat_InfreqTest_ind,'omitnan'),  sum(~isnan(lat_InfreqTest_ind)));

condPlots = {
    'Learning Frequent',   lat_FreqTrain_ind,   rt_train(:,1),  black
    'Learning Infrequent', lat_InfreqTrain_ind, rt_train(:,2),  black
    'Testing Frequent',    lat_FreqTest_ind,    rt_test(:,1),   black
    'Testing Infrequent',  lat_InfreqTest_ind,  rt_test(:,2),   black
};

tiColors = lines(4);
tiNames  = {'lamp', 'dress', 'chair', 'guitar'}; 

figure('Color','w','Position',[100 100 1050 800]);
for i = 1:4
    L = condPlots{i,2}; R = condPlots{i,3}; col = condPlots{i,4};
    ok = ~isnan(L) & ~isnan(R);
    [rP,pP] = corr(L(ok), R(ok), 'type','Pearson');

    subplot(2,2,i); hold on; set(gca,'FontSize',14,'LineWidth',1.5); box off;
    
     % scatter colored by frequent-target index (1-4)
    hLeg = gobjects(1,4);   % legend handles per targetIndex
    for ti = 1:4
        sel = ok & (targetIndex_ind == ti);
        if any(sel)
            hLeg(ti) = scatter(L(sel), R(sel), 50, tiColors(ti,:), ...
                'filled', 'MarkerFaceAlpha',0.8, 'DisplayName', tiNames{ti});
        end
    end

    % least-squares fit line
    b = polyfit(L(ok), R(ok), 1);
    xx = linspace(min(L(ok)), max(L(ok)), 100);
    plot(xx, polyval(b,xx), 'color', col, 'LineWidth', 2);
    xlabel('N2pc latency (ms)'); ylabel('RT (ms)');
    xlim([200 600]);     
    ylim([400 1000]);
    title(sprintf('%s\nPearson r=%.2f, p=%.3f', ...
        condPlots{i,1}, rP, pP), 'FontWeight','normal','FontSize',11);

    lgd = legend(hLeg(isgraphics(hLeg)));
    lgd.Title.String = 'Frequent target object';
    lgd.Title.FontWeight = 'normal';
    lgd.Position = [0.86 0.82 0.12 0.15];
    legend boxoff;
end
exportgraphics(gcf, 'N2pc_Latency_RT_correlation.png', 'Resolution', 300);

%% =====================================================================
%  PERMUTATION-BASED SPLIT-HALF RELIABILITY  (N2pc fractional-area latency
%  ). Spearman-Brown corrected, averaged over random
%  splits. Splits are stratified by target side so each half keeps the
%  contra/ipsi balance.
%% =====================================================================
nPerms = 1000;
rng(42);  % reproducible

% containers: rows = subjects, cols = perms
relCells = {  % phase, cond index, display name
    'train',1,'Train Frequent'
    'train',2,'Train Infrequent'
    'test', 1,'Test Frequent'
    'test', 2,'Test Infrequent'
};

% helper: build contra-ipsi diff wave from a subset of trials
buildDiff = @(eeg,LHc,RHc,isLeft,isRight) ...
    ( (mean(eeg(LHc,:,isLeft),3) + mean(eeg(RHc,:,isRight),3))/2 ) ...   % contra
  - ( (mean(eeg(LHc,:,isRight),3)+ mean(eeg(RHc,:,isLeft),3))/2 );       % ipsi

fprintf('\n=== Split-half reliability (%d permutations) ===\n', nPerms);

for rc = 1:size(relCells,1)
    ph   = relCells{rc,1}; % phase
    cIdx = relCells{rc,2}; % condition

    % per-perm, per-subject reliability inputs
    latA = nan(nsubs,nPerms);  latB = nan(nsubs,nPerms);

    for s = 1:nsubs
        D = SH.(ph){s,cIdx};
        eeg = D.eeg;
        isLeft = D.left(:);  isRight = D.right(:);
        nL = find(isLeft);   nR = find(isRight);

        for p = 1:nPerms
            % stratified split: halve left-target and right-target trials separately
            Lperm = nL(randperm(numel(nL)));
            Rperm = nR(randperm(numel(nR)));
            Lh = floor(numel(nL)/2);   Rh = floor(numel(nR)/2);

            % split half
            idxA = false(size(isLeft));  idxB = false(size(isLeft));
            idxA(Lperm(1:Lh))     = true;  idxA(Rperm(1:Rh))     = true;
            idxB(Lperm(Lh+1:2*Lh))= true;  idxB(Rperm(Rh+1:2*Rh))= true;

            % half A
            leftA  = isLeft  & idxA;  rightA = isRight & idxA;
            diffA  = buildDiff(eeg, LH, RH, leftA, rightA);
            wA = mean(diffA(idxElec,:),1);
            latA(s,p) = halfAreaLatency(wA, latWinBeg, latWinEnd, times);

            % half B
            leftB  = isLeft  & idxB;  rightB = isRight & idxB;
            diffB  = buildDiff(eeg, LH, RH, leftB, rightB);
            wB = mean(diffB(idxElec,:),1);
            latB(s,p) = halfAreaLatency(wB, latWinBeg, latWinEnd, times);
        end
    end

    % per-perm Spearman-Brown corrected reliability (Spearman corr), then summarize across perms
    relLat = nan(nPerms,1);  nUsed = nan(nPerms,1);
    for p = 1:nPerms
        okL = ~isnan(latA(:,p)) & ~isnan(latB(:,p));
        nUsed(p) = sum(okL);
        rL = corr(latA(okL,p), latB(okL,p), 'type','Spearman');
        relLat(p) = 2*rL/(1+rL);
    end

    fprintf('\n%s  (avg %.1f/%d subjects measurable per split)\n', ...
        relCells{rc,3}, mean(nUsed,'omitnan'), nsubs);
    fprintf('   Latency  reliability r_SB = %.3f\n', ...
        mean(relLat,'omitnan'));
end






