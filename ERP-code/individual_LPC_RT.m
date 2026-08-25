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

% Split by median, good half
% S.Subs = {'2', '12', '14', '16', '19', '20', '21', '24', '26', '29', '33', '38', '39', '40', '42', '44', '47', '48'};

% Split by median, poor half
% S.Subs = {'3', '7', '10', '12', '13', '15', '22', '23', '25', '30', '31', '34', '36', '37', '41', '45', '46', '49'};

nsubs = size(S.Subs,2);

% colors
darkblue  = [26, 128, 187]./255;
lightblue = [140, 197, 227]./255;
darkred   = [160, 0, 0]./255;
lightred  = [216, 166, 166]./255;
black = [0, 0, 0]./255;

S.BaselinBeg = -200;
S.BaselinEnd = 0;

startTime = -300; % ms
endTime   = 798;  % ms
numPoints = 550;
times = linspace(startTime, endTime, numPoints);

S.PlotTimeBegin = -200;
S.PlotTimeEnd   = 600;

% LPC mean-amplitude window
S.meanBegin = 400;
S.meanEnd   = 600;

%%% ------------------------------------------------------------------- %%%
% Channels for LPC (central/centro-parietal), averaged across these
[channelLabels, chanplot] = getElectrodeInfo();
whichChannels = {'CP5', 'CP6', 'Cz'};

% find electrode numbers
for i = 1:numel(whichChannels); idxElec(i) = find(strcmp(channelLabels, whichChannels(i)));end

%% ------------------------------------------------------------------- %%%
conds = [1 2]; % 1 = frequent, 2 = infrequent
condNames = {'Frequent', 'Infrequent'};

% per-subject ERPs (chan x time) and mean RTs
train_erp = cell(nsubs,2);
test_erp  = cell(nsubs,2);
rt_train  = nan(nsubs,2);
rt_test   = nan(nsubs,2);
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

    % vectorize
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

        %% --- Store averaged ERPs (chan x time) ---
        train_erp{SubNo, c} = mean(thisTrainEEG, 3);
        test_erp{SubNo, c}  = mean(thisTestEEG, 3);

        % --- per-subject mean RT, matched to kept trials + phase + condition ---
        trIdxC = trainIdx(targCondition(trainIdx) == conds(c));
        teIdxC = testIdx (targCondition(testIdx)  == conds(c));
        rt_train(SubNo,c) = mean(rt(trIdxC), 'omitnan');
        rt_test (SubNo,c) = mean(rt(teIdxC), 'omitnan');
    end
end

%% --- Time-window indices for LPC mean amplitude ---
[~, begMean] = min(abs(times - S.meanBegin));
[~, endMean] = min(abs(times - S.meanEnd));

%% --- Individual-subject LPC mean amplitudes (400-600 ms) ---
amp_FreqTrain_ind   = nan(nsubs,1);
amp_InfreqTrain_ind = nan(nsubs,1);
amp_FreqTest_ind    = nan(nsubs,1);
amp_InfreqTest_ind  = nan(nsubs,1);

for s = 1:nsubs
    % collapse across electrodes of interest, then mean over the window
    amp_FreqTrain_ind(s)   = mean(mean(train_erp{s,1}(idxElec, begMean:endMean), 2), 1);
    amp_InfreqTrain_ind(s) = mean(mean(train_erp{s,2}(idxElec, begMean:endMean), 2), 1);
    amp_FreqTest_ind(s)    = mean(mean(test_erp{s,1}(idxElec, begMean:endMean), 2), 1);
    amp_InfreqTest_ind(s)  = mean(mean(test_erp{s,2}(idxElec, begMean:endMean), 2), 1);
end

%% --- Export individual LPC amplitudes to CSV (long format) ---
phaseCol = {}; freqCol = {}; subjCol = {}; ampCol = []; rtCol = [];

condDef = {
    'Train', 'Frequent',   amp_FreqTrain_ind,   rt_train(:,1)
    'Train', 'Infrequent', amp_InfreqTrain_ind, rt_train(:,2)
    'Test',  'Frequent',   amp_FreqTest_ind,    rt_test(:,1)
    'Test',  'Infrequent', amp_InfreqTest_ind,  rt_test(:,2)
};

for i = 1:size(condDef,1)
    amp = condDef{i,3};
    rtv = condDef{i,4};
    for s = 1:nsubs
        subjCol{end+1,1}  = S.Subs{s};
        phaseCol{end+1,1} = condDef{i,1};
        freqCol{end+1,1}  = condDef{i,2};
        ampCol(end+1,1)   = amp(s);
        rtCol(end+1,1)    = rtv(s);
    end
end

T = table(subjCol, phaseCol, freqCol, ampCol, rtCol, ...
    'VariableNames', {'subject','phase','frequency','LPC_amplitude','RT'});
writetable(T, 'individual_LPC_amplitudes.csv');

% report
fprintf('\n--- Individual LPC mean amplitudes (mean +/- SD across subjects) ---\n');
fprintf('Learning Frequent:   %.2f +/- %.2f uV\n', mean(amp_FreqTrain_ind,'omitnan'),   std(amp_FreqTrain_ind,'omitnan'));
fprintf('Learning Infrequent: %.2f +/- %.2f uV\n', mean(amp_InfreqTrain_ind,'omitnan'), std(amp_InfreqTrain_ind,'omitnan'));
fprintf('Testing  Frequent:   %.2f +/- %.2f uV\n', mean(amp_FreqTest_ind,'omitnan'),    std(amp_FreqTest_ind,'omitnan'));
fprintf('Testing  Infrequent: %.2f +/- %.2f uV\n', mean(amp_InfreqTest_ind,'omitnan'),  std(amp_InfreqTest_ind,'omitnan'));

%% --- LPC amplitude x RT correlations + scatterplots ---
condPlots = {
    'Learning Frequent',   amp_FreqTrain_ind,   rt_train(:,1),  black
    'Learning Infrequent', amp_InfreqTrain_ind, rt_train(:,2),  black
    'Testing Frequent',    amp_FreqTest_ind,    rt_test(:,1),   black
    'Testing Infrequent',  amp_InfreqTest_ind,  rt_test(:,2),   black
};

tiColors = lines(4);
tiNames  = {'lamp', 'dress', 'chair', 'guitar'}; 

fprintf('\n--- LPC amplitude x RT correlations ---\n');
figure('Color','w','Position',[100 100 1050 800]);
for i = 1:4
    A = condPlots{i,2}; R = condPlots{i,3}; col = condPlots{i,4};
    ok = ~isnan(A) & ~isnan(R);
    [rP,pP] = corr(A(ok), R(ok), 'type','Pearson');

    fprintf('%s\nPearson r=%.2f, p=%.3f', ...
        condPlots{i,1}, rP, pP);
 
    subplot(2,2,i); hold on; set(gca,'FontSize',14,'LineWidth',1.5); box off;
 
    % scatter colored by frequent-target index (1-4)
    hLeg = gobjects(1,4);   % legend handles per targetIndex
    for ti = 1:4
        sel = ok & (targetIndex_ind == ti);
        if any(sel)
            hLeg(ti) = scatter(A(sel), R(sel), 50, tiColors(ti,:), ...
                'filled', 'MarkerFaceAlpha',0.8, 'DisplayName', tiNames{ti});
        end
    end
 
    % least-squares fit line (across all valid subjects, condition color)
    b = polyfit(A(ok), R(ok), 1);
    xx = linspace(min(A(ok)), max(A(ok)), 100);
    plot(xx, polyval(b,xx), 'color', col, 'LineWidth', 2, 'HandleVisibility','off');
 
    xlabel('LPC amplitude (\muV)'); ylabel('RT (ms)');
    xlim([-5 10]);     
    ylim([400 1000]);
    title(sprintf('%s\nPearson r=%.2f, p=%.3f', ...
        condPlots{i,1}, rP, pP), 'FontWeight','normal','FontSize',11);
 
    lgd = legend(hLeg(isgraphics(hLeg)));
    lgd.Title.String = 'Frequent target object';
    lgd.Title.FontWeight = 'normal';
    lgd.Position = [0.86 0.82 0.12 0.15];
    legend boxoff;
end
 
exportgraphics(gcf, 'LPC_amplitude_RT_correlation.png', 'Resolution', 300);