
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
close all
clc
clear all
clear global

S.homepath = pwd;
S.utils = 'utils';
S.EEGdatafolder='filteredEEG';
S.datafolder='BehavioralDataAligned';
S.pathsymbol = '/';

% Which eeg data file to load
S.eegDataFile='filtedEEG';
% S.erpDataFile = 'ERPfilt';
S.behaveFile = '_behavioral_data';

addpath('scripts');
addpath('utils');


% -------------------------------------------------------------------------
% Which subjects to load

% all
S.Subs = {'2', '3', '7', '10', '12', '13', '14', '15', '16', '19', '20', '21', '22', '23', '24', '25', '26', '29', '30', '31', '32', '33', '34', '36', '37', '38', '39', '40', '41', '42', '44', '45', '46', '47', '48', '49'};
% Split by median, good half
% S.Subs = {'2', '12', '14', '16', '19', '20', '21', '24', '26', '29', '33', '38', '39', '40', '42', '44', '47', '48'};

% Split by median, poor half
% S.Subs = {'3', '7', '10', '12', '13', '15', '22', '23', '25', '30', '31', '34', '36', '37', '41', '45', '46', '49'};
% S.Subs = {'44'}; 
nsubs = length(S.Subs);



%%---------------------------------------- %%%
% get channels
[channelLabels, chanplot] = getElectrodeInfo();

%%---------------------------------------- %%%
% which elecs to plot
% whichChannels  = {'PO7', 'P7', 'PO3', 'P3', 'PO8', 'P8', 'PO4', 'P4'}; % will average across these
whichChannels  = {'CP5', 'CP6', 'Cz'};


%% ---------------------------------------- %%%
% define some stuff
S.BaselinBeg = -200;
S.BaselinEnd = 0;
S.PlotTimeBegin = -200;
S.PlotTimeEnd = 600; % in ms
ylimits  = [-5 5];
S.meanBegin = 400;  
S.meanEnd   = 600;   %?

startTime = -300; % ms
endTime = 798;    % ms
numPoints = 550;
times = linspace(startTime, endTime, numPoints);


% which conditions
whichConditions1 = {'frequent train', 'infrequent train', 'frequent test', 'infrequent test'};

% colors
%colors
darkblue = [26, 128, 187]./255;
lightblue = [140, 197, 227]./255;
darkred = [160, 0, 0]./255;
lightred = [216, 166, 166]./255;

conds = [1 2]; % 1 = frequent, 2 = infrequent
condNames = {'Frequent', 'Infrequent'};

train_erp = cell(nsubs,2);
test_erp  = cell(nsubs,2);

% find electrodes numbers
for i = 1:numel(whichChannels); idxElec(i) = find(strcmp(channelLabels, whichChannels(i)));end

fprintf('Loading subjects and computing ERPs ...\n');
for SubNo = 1:size(S.Subs,2)
    disp(num2str(S.Subs{SubNo}));

    %----------------------------------------------------------------------
    % load eeg data
    % eLoad = load([S.homepath S.pathsymbol S.EEGdatafolder S.pathsymbol S.Subs{SubNo} S.eegDataFile]);
    % erpLoad = load([S.homepath S.pathsymbol S.EEGdatafolder S.pathsymbol S.Subs{SubNo} S.erpDataFile]);
    % e = eLoad.EEG;
    eLoad = load([S.homepath S.pathsymbol S.EEGdatafolder S.pathsymbol 'Subject_' S.Subs{SubNo} S.eegDataFile '.mat']);
    e = eLoad.eegKept;

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
    cutoff = floor(4/5 * nTrials); % keep fastest
    keepIdx = sortIdx(1:cutoff);   % indices of fastest trials
    
    % --- get trial indices ---
    trainIdx = keepIdx(blockNumber(keepIdx) <= 12);
    testIdx  = keepIdx(blockNumber(keepIdx) > 12);


    % Index EEG data using filtered RT
    %% --- Split train/test ---
    trainEEG = e(:,:,trainIdx);
    testEEG  = e(:,:,testIdx);

    trainCond = targCondition(trainIdx);
    testCond  = targCondition(testIdx);
    trainLoc  = targLoc(trainIdx);
    testLoc   = targLoc(testIdx);

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

        %% --- Store ERPs ---
        % Train
        train_erp{SubNo, c} = mean(thisTrainEEG, 3);
    
        % Test
        test_erp{SubNo, c} = mean(thisTestEEG, 3);
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
    Mean_FrequentTrain(SubNo) = squeeze(mean( ...
        mean(train_erp{SubNo,1}(idxElec, begMean:endMean), 2), 1));

    Mean_InfrequentTrain(SubNo) = squeeze(mean( ...
        mean(train_erp{SubNo,2}(idxElec, begMean:endMean), 2), 1));
    
    Mean_FrequentTest(SubNo) = squeeze(mean( ...
        mean(test_erp{SubNo,1}(idxElec, begMean:endMean), 2), 1));
    
    Mean_InfrequentTest(SubNo) = squeeze(mean( ...
        mean(test_erp{SubNo,2}(idxElec, begMean:endMean), 2), 1));


end


% Plot ERP for each individual
%-------------------------------------------------------------------------
%% Training
figure; hold on;
set(gca, 'FontSize', 18, 'LineWidth', 3);
if nsubs < 1
    % --- Plot for one subject ---
    plot(time_range, mean(train_erp{2}(:,idxElec,:),2), 'color',darkred, 'LineWidth', 2); % infrequent
    plot(time_range, mean(train_erp{1}(:,idxElec,:),2), 'color',darkblue, 'LineWidth', 2); % frequent
    % plot(e.times, mean(test_erp{1}(:,idxElec,:),2), 'color',darkred, 'LineWidth', 2);
    % plot(e.times, mean(test_erp{2}(:,idxElec,:),2), 'color',lightred, 'LineWidth', 2);

else
    % --- Compute grand average across subjects ---
    fprintf('Grand averaging ...\n');

    grandTrain = zeros(2, numel(times));
    % grandTest  = zeros(2, numel(e.times));

    for c = 1:2   % 1 = frequent, 2 = infrequent
        for s = 1:nsubs
            % Average across channels of interest for this subject/condition
            subjTrain = mean(train_erp{s, c}(idxElec,:), 1); % 1 × time
            % subjTest  = mean(test_erp{s, c}(idxElec,:), 1);  % 1 × time

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
    yline(0, 'k', 'LineWidth', 1, 'HandleVisibility', 'off');
    xline(0, 'k', 'LineWidth', 1, 'HandleVisibility', 'off');
    yticks([-2,-1,0,1, 2, 3, 4]);
    trainFreq = squeeze(grandTrain(1,:,:));
    trainInfreq = squeeze(grandTrain(2,:,:));
    plot(time_range, trainInfreq(plotBeg:plotEnd), 'color', darkred, 'LineWidth', 2)
    plot(time_range, trainFreq(plotBeg:plotEnd), 'color', darkblue, 'LineWidth', 2);
    % plot(e.times, squeeze(grandTest(1,:,:)), 'color', darkred, 'LineWidth', 2);
    % plot(e.times, squeeze(grandTest(2,:,:)), 'color', lightred, 'LineWidth', 2);
end

xlabel('Time (ms)'), ylabel(' µV'); hold on;
p1 = plot(time_range, trainInfreq(plotBeg:plotEnd), 'color', darkred,  'LineWidth', 2);
p2 = plot(time_range, trainFreq(plotBeg:plotEnd),   'color', darkblue, 'LineWidth', 2);
lg = legend([p2, p1], {'Frequent', 'Infrequent'});
legend boxoff;

% Get the current figure handle
f = gcf;

% Export the figure as a 300 DPI PNG file
exportgraphics(f, 'LPC_good_train_80.png', 'Resolution', 300);


%% Testing
figure; hold on;
set(gca, 'FontSize', 18, 'LineWidth', 3);


if nsubs < 1
    % --- Plot for one subject ---
    % plot(e.times, mean(train_erp{1}(:,idxElec,:),2), 'color',darkblue, 'LineWidth', 2); % frequent
    % plot(e.times, mean(train_erp{2}(:,idxElec,:),2), 'color',lightblue, 'LineWidth', 2); % infrequent
    plot(time_range, mean(test_erp{2}(:,idxElec,:),2), 'color',lightred, 'LineWidth', 2);
    plot(time_range, mean(test_erp{1}(:,idxElec,:),2), 'color',lightblue, 'LineWidth', 2);

else
    % --- Compute grand average across subjects ---
    fprintf('Grand averaging ...\n');

    % grandTrain = zeros(2, numel(e.times));
    grandTest  = zeros(2, numel(times));

    for c = 1:2   % 1 = frequent, 2 = infrequent
        for s = 1:nsubs
            % Average across channels of interest for this subject/condition
            % subjTrain = mean(train_erp{s, c}(idxElec,:), 1); % 1 × time
            subjTest  = mean(test_erp{s, c}(idxElec,:), 1);  % 1 × time

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
    % plot(e.times, squeeze(grandTrain(1,:,:)), 'color', darkblue, 'LineWidth', 2);
    % plot(e.times, squeeze(grandTrain(2,:,:)), 'color', lightblue, 'LineWidth', 2);
    yticks([-2,-1,0,1, 2, 3, 4]);
    testFreq = squeeze(grandTest(1,:,:));
    testInfreq = squeeze(grandTest(2,:,:));
    plot(time_range, testInfreq(plotBeg:plotEnd), 'color', lightred, 'LineWidth', 2);
    plot(time_range, testFreq(plotBeg:plotEnd), 'color', lightblue, 'LineWidth', 2);
end

xlabel('Time (ms)'), ylabel(' µV'); hold on;
p1 = plot(time_range, testInfreq(plotBeg:plotEnd), 'color', lightred,  'LineWidth', 2);
p2 = plot(time_range, testFreq(plotBeg:plotEnd),   'color', lightblue, 'LineWidth', 2);
lg = legend([p2, p1], {'Frequent', 'Infrequent'});
legend boxoff;

% Get the current figure handle
f = gcf;

% Export the figure as a 300 DPI PNG file
exportgraphics(f, 'LPC_good_test_80.png', 'Resolution', 300);




% plot mean amplitudes
figure(); clf;
set(gca, 'FontSize', 24, 'LineWidth', 3);hold on;
title('Mean amplitude', 'FontWeight', 'normal');
barSemWithin({Mean_FrequentTrain, Mean_InfrequentTrain, Mean_FrequentTest, Mean_InfrequentTest}, ...
    {'train freq', 'train infreq', 'test freq', 'test infreq'}, {darkblue, darkred, lightblue, lightred}); hold on;

% Get the current figure handle
f = gcf;

% Export the figure as a 300 DPI PNG file
exportgraphics(f, 'LPC_good_amp_80.png', 'Resolution', 300);

% Means
mFreqTrain = mean(Mean_FrequentTrain);
mInfreqTrain = mean(Mean_InfrequentTrain);
mFreqTest = mean(Mean_FrequentTest);
mInfreqTest = mean(Mean_InfrequentTest);

% Standard deviations
sdFreqTrain = std(Mean_FrequentTrain);
sdInfreqTrain = std(Mean_InfrequentTrain);
sdFreqTest = std(Mean_FrequentTest);
sdInfreqTest = std(Mean_InfrequentTest);

% ANOVA
% data = [Mean_FrequentTrain(:), ...
%         Mean_InfrequentTrain(:), ...
%         Mean_FrequentTest(:), ...
%         Mean_InfrequentTest(:)];
% nSubs = length(S.Subs);
% 
% % Put into a table (required format for fitrm)
% T = table((1:nSubs)', data(:,1), data(:,2), data(:,3), data(:,4), ...
%     'VariableNames', {'Subject','FreqTrain','InfreqTrain','FreqTest','InfreqTest'});
% 
% % Define repeated measures model
% rm = fitrm(T, 'FreqTrain-InfreqTest ~ 1', 'WithinDesign', ...
%     table([1 1 2 2]', [1 2 1 2]', 'VariableNames', {'Phase','Frequency'}));
% 
% % Run repeated-measures ANOVA
% ranovatbl = ranova(rm, 'WithinModel','Phase*Frequency');
% disp(ranovatbl);
% 
% disp(multcompare(rm,'Frequency','By','Phase'));  % Simple effects
% 

data = [Mean_FrequentTrain(:), ...
        Mean_InfrequentTrain(:), ...
        Mean_FrequentTest(:), ...
        Mean_InfrequentTest(:)];
nSubs = length(S.Subs);

%% --- Reshape Data for ANOVAN (Wide to Long Format) ---

% Vector of all measurements (stacking the columns of 'data')
y = data(:); 

% Create the grouping variable for the "Phase" factor (Train, Test)
% The first 2*nSubs values are 'Train', the next 2*nSubs are 'Test'
phase_group = [repmat({'Train'}, 2 * nSubs, 1); ...
               repmat({'Test'},  2 * nSubs, 1)];

% Create the grouping variable for the "Frequency" factor (Frequent, Infrequent)
% This alternates every nSubs values
freq_group = repmat([repmat({'Frequent'},   nSubs, 1); ...
                     repmat({'Infrequent'}, nSubs, 1)], 2, 1);

% Create the grouping variable for Subjects
% This is the random factor.
subject_group = repmat((1:nSubs)', 4, 1);

%% --- Run the ANOVA using ANOVAN ---
[p, tbl, stats] = anovan(y, {subject_group, phase_group, freq_group}, ...
    'model', 'interaction', ...
    'random', 1, ...
    'varnames', {'Subject', 'Phase', 'Frequency'});

disp('--- ANOVAN Results ---');
disp(tbl);

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


disp('--- Post-Hoc Comparisons for the Phase*Frequency Interaction ---');

% Paired Ttest
[hTrain, pTrain, ciTrain, statsTrain] = ttest(Mean_FrequentTrain, Mean_InfrequentTrain);
fprintf('t(%d) = %.3f, p = %.4f\n', statsTrain.df, statsTrain.tstat, pTrain);
[hTest, pTest, ciTest, statsTest] = ttest(Mean_FrequentTest, Mean_InfrequentTest);
fprintf('t(%d) = %.3f, p = %.4f\n', statsTest.df, statsTest.tstat, pTest);

%% =====================================================================
%  LPC raw-voltage TOPOGRAPHY, 400-600 ms, per condition & phase
%  Option A. Requires EEGLAB on path (run `eeglab nogui` first).
%  Reuses train_erp / test_erp (chan x time), already built in the loop.
%  Run AFTER the subject loop (e.g. after line 333). No loop edits needed.
%% =====================================================================

%% ---- 1. Channel locations ----
% Uses the same montage file as the N2pc script. Confirm channelLabels
% order matches the .loc order (row 1 = FP1, etc.).
chanlocs = readlocs(fullfile(S.utils, '32channelsGreenwithDiode.loc'));
labels   = {chanlocs.labels};
nChan    = numel(chanlocs);

% Channels to exclude from the scalp plot (non-scalp / no pair)
nonScalp = find(ismember(upper(labels), {'M1','M2','HEOG','VEOG','DIODE'}));

%% ---- 2. Time window 400-600 ms ----
[~, tBeg] = min(abs(times - 400));
[~, tEnd] = min(abs(times - 600));

%% ---- 3. Grand-average scalp map per phase/condition ----
condNames  = {'Frequent','Infrequent'};
phaseNames = {'Train','Test'};

figure('Color','w','Name','LPC topography 400-600ms', ...
       'Position',[100 100 900 800]);
plotN = 1;

for phase = 1:2
    if phase==1, src = train_erp; else, src = test_erp; end
    for c = 1:2
        M = zeros(nChan, nsubs);
        for s = 1:nsubs
            M(:,s) = mean(src{s,c}(:, tBeg:tEnd), 2);   % chan x 1, mean over window
        end
        scalp = mean(M, 2);                             % grand average over subjects
        scalp(nonScalp) = 0;                            % suppress non-scalp chans

        subplot(2,2,plotN); plotN = plotN+1;
        topoplot(scalp, chanlocs, ...
                 'maplimits', [-4 4], ...               % adjust to your uV range
                 'electrodes','on', 'style','map', ...
                 'plotchans', setdiff(1:nChan, nonScalp));
        title(sprintf('%s - %s', phaseNames{phase}, condNames{c}));
        colorbar;
    end
end

exportgraphics(gcf, 'LPC_topo_poor.png', 'Resolution', 300);