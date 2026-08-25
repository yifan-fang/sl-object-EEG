clc
clear all
clear global
close all;

addpath('scripts');
addpath('utils');

S.homepath = pwd;
S.utils = 'utils';
S.EEGdatafolder='filteredEEG';
S.datafolder='BehavioralDataAligned';
S.pathsymbol = '/';

% Which eeg data file to load
S.eegDataFile='filtedEEG';
S.behaveFile = '_behavioral_data';

% Exp 1 (all)
S.Subs = {'2', '3', '7', '10', '12', '13', '14', '15', '16', '19', '20', '21', '22', '23', '24', '25', '26', '29', '30', '31', '32', '33', '34', '36', '37', '38', '39', '40', '41', '42', '44', '45', '46', '47', '48', '49'};

nsubs = size(S.Subs,2);

% colors
black = [0, 0, 0]./255;

S.BaselinBeg = -200;
S.BaselinEnd = 0;

startTime = -300; % ms
endTime   = 798;  % ms
numPoints = 550;
times = linspace(startTime, endTime, numPoints);

% N2pc fractional-area latency window
latencyWindow = [175, 600];   % ms
% LPC mean-amplitude window
S.lpcBegin = 400;             % ms
S.lpcEnd   = 600;             % ms

%%% ------------------------------------------------------------------- %%%
% N2pc electrodes (lateral occipital, contra-ipsi difference wave)
[LH, RH, flippedChannels] = ipsiconElectrodes();
S.IpsiconChannels = flippedChannels;
whichChannelsN2pc = {'PO7/8', 'P7/8'};
for i = 1:numel(whichChannelsN2pc)
    idxElecN2pc(i) = find(strcmp(S.IpsiconChannels, whichChannelsN2pc{i}));
end

% LPC electrodes (central / centro-parietal, condition-average ERP)
[channelLabels, chanplot] = getElectrodeInfo();
whichChannelsLPC = {'CP5', 'CP6', 'Cz'};
for i = 1:numel(whichChannelsLPC)
    idxElecLPC(i) = find(strcmp(channelLabels, whichChannelsLPC{i}));
end

%% ------------------------------------------------------------------- %%%
conds = [1 2]; % 1 = frequent, 2 = infrequent

% per-subject containers
train_ic_diff = cell(nsubs,2);   % N2pc contra-ipsi diff (chan x time)
test_ic_diff  = cell(nsubs,2);
train_erp     = cell(nsubs,2);   % LPC condition-average ERP (chan x time)
test_erp      = cell(nsubs,2);
targetIndex_ind = nan(nsubs,1);

fprintf('Loading subjects and computing ERPs ...\n');
for SubNo = 1:size(S.Subs,2)
    disp(num2str(S.Subs{SubNo}));

    %----------------------------------------------------------------------
    % load eeg data
    eLoad = load([S.homepath S.pathsymbol S.EEGdatafolder S.pathsymbol 'Subject_' S.Subs{SubNo} S.eegDataFile '.mat']);
    e = eLoad.eegKept;

    %----------------------------------------------------------------------
    % Load behavior and consolidate into vectors:
    d = load([S.homepath S.pathsymbol S.datafolder S.pathsymbol 'Subject_' S.Subs{SubNo} S.behaveFile '.mat']);

    ID          = [d.data_table.ID];
    trialNumber = [d.data_table.TrialNumber];
    blockNumber = [d.data_table.BlockNumber];
    targID      = [d.data_table.TargetID];
    targLoc     = [d.data_table.TargetLocation];
    targCondition = [d.data_table.TargetCondition];
    acc = [d.data_table.Accuracy];
    rt  = [d.data_table.RT];

    % --- drop slowest rt (keep fastest 4/5) ---
    nTrials = numel(rt);
    [~, sortIdx] = sort(rt);
    cutoff  = floor(4/5 * nTrials);
    keepIdx = sortIdx(1:cutoff);

    % --- get trial indices ---
    trainIdx = keepIdx(blockNumber(keepIdx) <= 12);
    testIdx  = keepIdx(blockNumber(keepIdx) > 12);

    %% --- Split train/test ---
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
        %% --- Train ERP (baseline corrected trial-by-trial) ---
        condTrainIdx = find(trainCond == conds(c));
        thisTrainEEG = trainEEG(:,:,condTrainIdx);   % chan x time x nTrials
        for tr = 1:size(thisTrainEEG,3)
            baselineMean = mean(thisTrainEEG(:,begBaseline:endBaseline,tr), 2);
            thisTrainEEG(:,:,tr) = thisTrainEEG(:,:,tr) - baselineMean;
        end

        %% --- Test ERP (baseline corrected trial-by-trial) ---
        condTestIdx = find(testCond == conds(c));
        thisTestEEG = testEEG(:,:,condTestIdx);
        for tr = 1:size(thisTestEEG,3)
            baselineMean = mean(thisTestEEG(:,begBaseline:endBaseline,tr), 2);
            thisTestEEG(:,:,tr) = thisTestEEG(:,:,tr) - baselineMean;
        end

        %% --- N2pc: contra / ipsi difference wave ---
        % Train
        leftTrials  = trainLoc(condTrainIdx) < 3;
        rightTrials = trainLoc(condTrainIdx) > 2;
        train_ipsi   = (mean(thisTrainEEG(LH,:,leftTrials),3)  + mean(thisTrainEEG(RH,:,rightTrials),3)) / 2;
        train_contra = (mean(thisTrainEEG(LH,:,rightTrials),3) + mean(thisTrainEEG(RH,:,leftTrials),3))  / 2;
        train_ic_diff{SubNo, c} = train_contra - train_ipsi;

        % Test
        leftTrials  = testLoc(condTestIdx) < 3;
        rightTrials = testLoc(condTestIdx) > 2;
        test_ipsi   = (mean(thisTestEEG(LH,:,leftTrials),3)  + mean(thisTestEEG(RH,:,rightTrials),3)) / 2;
        test_contra = (mean(thisTestEEG(LH,:,rightTrials),3) + mean(thisTestEEG(RH,:,leftTrials),3))  / 2;
        test_ic_diff{SubNo, c} = test_contra - test_ipsi;

        %% --- LPC: condition-average ERP (chan x time) ---
        train_erp{SubNo, c} = mean(thisTrainEEG, 3);
        test_erp{SubNo, c}  = mean(thisTestEEG, 3);
    end
end

%% --- Window indices ---
[~, latWinBeg] = min(abs(times - latencyWindow(1)));
[~, latWinEnd] = min(abs(times - latencyWindow(2)));
[~, begMean]   = min(abs(times - S.lpcBegin));
[~, endMean]   = min(abs(times - S.lpcEnd));

%% --- Individual-subject N2pc latencies (50% fractional area) ---
lat_FreqTrain_ind   = nan(nsubs,1);
lat_InfreqTrain_ind = nan(nsubs,1);
lat_FreqTest_ind    = nan(nsubs,1);
lat_InfreqTest_ind  = nan(nsubs,1);

%% --- Individual-subject LPC mean amplitudes (400-600 ms) ---
amp_FreqTrain_ind   = nan(nsubs,1);
amp_InfreqTrain_ind = nan(nsubs,1);
amp_FreqTest_ind    = nan(nsubs,1);
amp_InfreqTest_ind  = nan(nsubs,1);

for s = 1:nsubs
    % N2pc: collapse over occipital electrodes -> 1 x time, then fractional-area latency
    wTrainFreq   = mean(train_ic_diff{s,1}(idxElecN2pc,:), 1);
    wTrainInfreq = mean(train_ic_diff{s,2}(idxElecN2pc,:), 1);
    wTestFreq    = mean(test_ic_diff{s,1}(idxElecN2pc,:),  1);
    wTestInfreq  = mean(test_ic_diff{s,2}(idxElecN2pc,:),  1);

    lat_FreqTrain_ind(s)   = halfAreaLatency(wTrainFreq,   latWinBeg, latWinEnd, times);
    lat_InfreqTrain_ind(s) = halfAreaLatency(wTrainInfreq, latWinBeg, latWinEnd, times);
    lat_FreqTest_ind(s)    = halfAreaLatency(wTestFreq,    latWinBeg, latWinEnd, times);
    lat_InfreqTest_ind(s)  = halfAreaLatency(wTestInfreq,  latWinBeg, latWinEnd, times);

    % LPC: collapse over central electrodes, then mean over window
    amp_FreqTrain_ind(s)   = mean(mean(train_erp{s,1}(idxElecLPC, begMean:endMean), 2), 1);
    amp_InfreqTrain_ind(s) = mean(mean(train_erp{s,2}(idxElecLPC, begMean:endMean), 2), 1);
    amp_FreqTest_ind(s)    = mean(mean(test_erp{s,1}(idxElecLPC,  begMean:endMean), 2), 1);
    amp_InfreqTest_ind(s)  = mean(mean(test_erp{s,2}(idxElecLPC,  begMean:endMean), 2), 1);
end


%% --- N2pc latency x LPC amplitude correlations + scatterplots ---
condPlots = {
    'Learning Frequent',   lat_FreqTrain_ind,   amp_FreqTrain_ind,   black
    'Learning Infrequent', lat_InfreqTrain_ind, amp_InfreqTrain_ind, black
    'Testing Frequent',    lat_FreqTest_ind,    amp_FreqTest_ind,    black
    'Testing Infrequent',  lat_InfreqTest_ind,  amp_InfreqTest_ind,  black
};

tiColors = lines(4);
tiNames  = {'lamp', 'dress', 'chair', 'guitar'};

fprintf('\n--- N2pc latency x LPC amplitude correlations ---\n');
figure('Color','w','Position',[100 100 1050 800]);
for i = 1:4
    L = condPlots{i,2};    % N2pc latency (x)
    A = condPlots{i,3};    % LPC amplitude (y)
    col = condPlots{i,4};
    ok = ~isnan(L) & ~isnan(A);
    [rP,pP] = corr(L(ok), A(ok), 'type','Pearson');

    fprintf('%s\nPearson r=%.2f, p=%.3f  (n=%d)\n', condPlots{i,1}, rP, pP, sum(ok));

    subplot(2,2,i); hold on; set(gca,'FontSize',14,'LineWidth',1.5); box off;

    % scatter colored by frequent-target index (1-4)
    hLeg = gobjects(1,4);
    for ti = 1:4
        sel = ok & (targetIndex_ind == ti);
        if any(sel)
            hLeg(ti) = scatter(L(sel), A(sel), 50, tiColors(ti,:), ...
                'filled', 'MarkerFaceAlpha',0.8, 'DisplayName', tiNames{ti});
        end
    end

    % least-squares fit line (across all valid subjects)
    b = polyfit(L(ok), A(ok), 1);
    xx = linspace(min(L(ok)), max(L(ok)), 100);
    plot(xx, polyval(b,xx), 'color', col, 'LineWidth', 2, 'HandleVisibility','off');

    xlabel('N2pc latency (ms)'); ylabel('LPC amplitude (\muV)');
    xlim([200 600]);
    ylim([-5 10]);
    title(sprintf('%s\nPearson r=%.2f, p=%.3f', ...
        condPlots{i,1}, rP, pP), 'FontWeight','normal','FontSize',11);

    lgd = legend(hLeg(isgraphics(hLeg)));
    lgd.Title.String = 'Frequent target object';
    lgd.Title.FontWeight = 'normal';
    lgd.Position = [0.86 0.82 0.12 0.15];
    legend boxoff;
end

exportgraphics(gcf, 'N2pc_latency_LPC_amplitude_correlation.png', 'Resolution', 300);

%% ----------------------------------------------------------------------
% Local function: 50% fractional-area latency on the contra-ipsi diff wave
%% ----------------------------------------------------------------------
function latMS = halfAreaLatency(wave, winBeg, winEnd, times)
    segment = wave(winBeg:winEnd);
    segment_pos = segment * -1;          % flip so N2pc is positive
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