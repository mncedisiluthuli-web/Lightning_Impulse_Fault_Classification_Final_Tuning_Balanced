% Master's in Electrical and Electronics Engineering (CW)
% Transformer Lightning Impulse Fault Classification
% MEE9X02 - Minor Dissertation
% Mncedisi Luthuli 225251946

clear; clc;
rng(42);

%% === Parameters ===
folder = 'C:\Users\Mnced\OneDrive\Documents\MATLAB';
files  = dir(fullfile(folder, 'TX*.xlsx'));

%% === Data Loading and Feature Extraction ===
feature_data = cell(length(files)*3*4, 10); % (file,unit,phase,wave,label, T1,T2,Vp,Ip,Beta)
row = 0;

for f = 1:length(files)
    filePath = fullfile(files(f).folder, files(f).name);
    data = readtable(filePath);
    t    = toNumericArray(data.Time);
    unit = data.Unit(1);

    for p = 1:3
        switch p
            case 1
                phase = 'A'; v60 = 'Voltage_60_1'; c60 = 'Current_60_1';
                v100 = {'Voltage_100_1', 'Voltage_100_2', 'Voltage_100_3'};
                c100 = {'Current_100_1', 'Current_100_2', 'Current_100_3'};
                label_col = 'Fault_Condition_A_phase';
            case 2
                phase = 'B'; v60 = 'Voltage_60_2'; c60 = 'Current_60_2';
                v100 = {'Voltage_100_1_2', 'Voltage_100_2_2', 'Voltage_100_3_2'};
                c100 = {'Current_100_1_2', 'Current_100_2_2', 'Current_100_3_2'};
                label_col = 'Fault_Condition_B_phase';
            case 3
                phase = 'C'; v60 = 'Voltage_60_3'; c60 = 'Current_60_3';
                v100 = {'Voltage_100_1_3', 'Voltage_100_2_3', 'Voltage_100_3_3'};
                c100 = {'Current_100_1_3', 'Current_100_2_3', 'Current_100_3_3'};
                label_col = 'Fault_Condition_C_phase';
        end

        label = string(data.(label_col){1});

        % --- 60% waveform
        if ismember(v60, data.Properties.VariableNames) && ismember(c60, data.Properties.VariableNames)
            val_v60  = toNumericArray(data.(v60));
            val_c60  = toNumericArray(data.(c60));
            val_v60  = fillmissing(val_v60, 'linear', 'EndValues', 'nearest');
            val_c60  = fillmissing(val_c60, 'linear', 'EndValues', 'nearest');
            t_filled = fillmissing(t,      'linear', 'EndValues', 'nearest');

            if all(~isnan(val_v60)) && all(~isnan(val_c60))
                feats = extract_waveform_features(t_filled, val_v60, val_c60);
                if all(~isnan(cell2mat(feats)))
                    row = row + 1;
                    feature_data(row, :) = [{files(f).name},{unit},{phase},{'60'},{label}, feats{:}];
                end
            end
        end

        % --- 100% waveforms
        for i100 = 1:3
            vcol = v100{i100};
            ccol = c100{i100};
            if ismember(vcol, data.Properties.VariableNames) && ismember(ccol, data.Properties.VariableNames)
                val_v100 = toNumericArray(data.(vcol));
                val_c100 = toNumericArray(data.(ccol));
                val_v100 = fillmissing(val_v100, 'linear', 'EndValues', 'nearest');
                val_c100 = fillmissing(val_c100, 'linear', 'EndValues', 'nearest');
                t_filled = fillmissing(t,      'linear', 'EndValues', 'nearest');

                if all(~isnan(val_v100)) && all(~isnan(val_c100))
                    feats = extract_waveform_features(t_filled, val_v100, val_c100);
                    if all(~isnan(cell2mat(feats)))
                        row = row + 1;
                        wave_type = ['100_' num2str(i100)];
                        feature_data(row, :) = [{files(f).name},{unit},{phase},{wave_type},{label}, feats{:}];
                    end
                end
            end
        end
    end
end

feature_data = feature_data(1:row, :);
T = cell2table(feature_data, 'VariableNames', ...
    {'File','Unit','Phase','WaveType','Label', ...
    'T1_voltage','T2_voltage','Vp_voltage','Ip_current','Beta_voltage'});

T = rmmissing(T);

outFile = fullfile(folder, 'Features_Table_Final.xlsx');  % fixed name
writetable(T, outFile);
disp(['✅ Features saved to: ' outFile]);

%% === Robust Average Overlay Plots (consistent V/I shading, no AGROW) ===
disp('Running robust waveform analysis and overlays...');

healthyLabel = 'Good';
faultyLabel  = 'Fault';
colH = [0 0.6 0];    % green
colF = [0.85 0 0];   % red
phases    = {'A','B','C'};
waveforms = {'60','100_1','100_2','100_3'};

for p = 1:numel(phases)
    phase = phases{p};
    figure('Name', sprintf('Phase %s - Average Overlays', phase), 'NumberTitle','off');
    tl = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
    title(tl, sprintf('Phase %s - Healthy vs Faulty Average Waveforms', phase));

    for w = 1:numel(waveforms)
        waveType = waveforms{w};
        nexttile; hold on; grid on;

        waveData = struct(); % HV,FV,HI,FI
        timeVec  = [];       % master time per subplot

        % --- loop Healthy / Faulty ----------------------------------------------
        for labelType = ["Healthy","Faulty"]
            lbl  = healthyLabel; cV = colH; cI = colH; tagV = 'HV'; tagI = 'HI';
            if labelType=="Faulty", lbl = faultyLabel; cV = colF; cI = colF; tagV = 'FV'; tagI = 'FI'; end

            idx = strcmp(T.Phase,phase) & strcmp(T.WaveType,waveType) & strcmp(T.Label,lbl);
            filesList = unique(T.File(idx));
            nFiles = numel(filesList);
            if nFiles==0, continue; end

            % Decide column names once
            switch phase
                case 'A'
                    switch waveType
                        case '60',    vCol='Voltage_60_1';    cCol='Current_60_1';
                        case '100_1', vCol='Voltage_100_1';   cCol='Current_100_1';
                        case '100_2', vCol='Voltage_100_2';   cCol='Current_100_2';
                        case '100_3', vCol='Voltage_100_3';   cCol='Current_100_3';
                    end
                case 'B'
                    switch waveType
                        case '60',    vCol='Voltage_60_2';    cCol='Current_60_2';
                        case '100_1', vCol='Voltage_100_1_2'; cCol='Current_100_1_2';
                        case '100_2', vCol='Voltage_100_2_2'; cCol='Current_100_2_2';
                        case '100_3', vCol='Voltage_100_3_2'; cCol='Current_100_3_2';
                    end
                case 'C'
                    switch waveType
                        case '60',    vCol='Voltage_60_3';    cCol='Current_60_3';
                        case '100_1', vCol='Voltage_100_1_3'; cCol='Current_100_1_3';
                        case '100_2', vCol='Voltage_100_2_3'; cCol='Current_100_2_3';
                        case '100_3', vCol='Voltage_100_3_3'; cCol='Current_100_3_3';
                    end
            end  % <-- end switch phase

            % 1) reference time & preallocate (outside per-file loop)
            timeVec = [];
            for ff0 = 1:nFiles
                data0 = readtable(fullfile(folder, filesList{ff0}));
                if ~ismember(vCol,data0.Properties.VariableNames) || ~ismember(cCol,data0.Properties.VariableNames), continue; end
                t0 = fillmissing(toNumericArray(data0.Time),'linear','EndValues','nearest');
                if ~isempty(t0), timeVec = t0(:); break; end
            end
            if isempty(timeVec), warning('No valid time for %s %s %s', phase, waveType, lbl); continue; end

            nS   = numel(timeVec);
            allV = nan(nFiles, nS);
            allI = nan(nFiles, nS);
            allT1 = nan(1, nFiles);
            allT2 = nan(1, nFiles);

            % 2) populate
            for ff = 1:nFiles
                data = readtable(fullfile(folder, filesList{ff}));
                if ~ismember(vCol,data.Properties.VariableNames) || ~ismember(cCol,data.Properties.VariableNames), continue; end

                tvec = fillmissing(toNumericArray(data.Time),'linear','EndValues','nearest');
                v = normalize_signal(fillmissing(toNumericArray(data.(vCol)),'linear','EndValues','nearest'));
                i = normalize_signal(fillmissing(toNumericArray(data.(cCol)),'linear','EndValues','nearest'));
                v = interp1(tvec(:), v(:), timeVec, 'linear','extrap').';
                i = interp1(tvec(:), i(:), timeVec, 'linear','extrap').';
                allV(ff,:) = v;  allI(ff,:) = i;

                [~, idxVp] = max(v);
                afterPk = find(timeVec > timeVec(idxVp));
                if ~isempty(afterPk)
                    iT1 = afterPk(find(v(afterPk) <= 0.1*v(idxVp), 1, 'first'));
                    iT2 = afterPk(find(v(afterPk) <= 0.5*v(idxVp), 1, 'first'));
                    if ~isempty(iT1), allT1(ff) = timeVec(iT1); end
                    if ~isempty(iT2), allT2(ff) = timeVec(iT2); end
                end
            end  % <-- end for ff

            % aggregate
            if any(~isnan(allV(:)))
                waveData.(tagV).mu  = mean(allV,1,'omitnan');
                waveData.(tagV).sd  = std(allV,0,1,'omitnan');
                waveData.(tagV).clr = cV;
                waveData.(tagV).T1  = mean(allT1,'omitnan');
                waveData.(tagV).T2  = mean(allT2,'omitnan');
            end
            if any(~isnan(allI(:))) && any(std(allI,0,1,'omitnan')>1e-6,'all')
                waveData.(tagI).mu  = mean(allI,1,'omitnan');
                waveData.(tagI).sd  = std(allI,0,1,'omitnan');
                waveData.(tagI).clr = cI;
            end
        end  % <-- end for labelType
        % ------------------------------------------------------------------------


        if isempty(timeVec)
            warning('No valid time for Phase %s, WaveType %s', phase, waveType);
            continue;
        end

        % ---------- Voltage (left axis) ----------
        yyaxis left; set(gca,'YColor',colH);
        h = gobjects(1,4);  % preallocate handles
        li = 0;             % legend index

        if isfield(waveData,'HV')
            shaded_mean(timeVec, waveData.HV.mu, waveData.HV.sd, waveData.HV.clr, 0.15);
            h1 = plot(timeVec, waveData.HV.mu, '-',  'Color', waveData.HV.clr, 'LineWidth',2, ...
                'DisplayName','Healthy Voltage');
            li=li+1; h(li)=h1;
        end
        if isfield(waveData,'FV')
            shaded_mean(timeVec, waveData.FV.mu, waveData.FV.sd, waveData.FV.clr, 0.15);
            h2 = plot(timeVec, waveData.FV.mu, '--', 'Color', waveData.FV.clr, 'LineWidth',2, ...
                'DisplayName','Faulty Voltage');
            li=li+1; h(li)=h2;
        end
        ylabel('Normalized Voltage'); ylim([-1.2 0.8]);

        % ---------- Current (right axis) ----------
        yyaxis right; set(gca,'YColor',colF);
        if isfield(waveData,'HI')
            shaded_mean(timeVec, waveData.HI.mu, waveData.HI.sd, waveData.HI.clr, 0.10);
            h3 = plot(timeVec, waveData.HI.mu, '-',  'Color', waveData.HI.clr, 'LineWidth',2, ...
                'DisplayName','Healthy Current');
            li=li+1; h(li)=h3;
        end
        if isfield(waveData,'FI')
            shaded_mean(timeVec, waveData.FI.mu, waveData.FI.sd, waveData.FI.clr, 0.10);
            h4 = plot(timeVec, waveData.FI.mu, '--', 'Color', waveData.FI.clr, 'LineWidth',2, ...
                'DisplayName','Faulty Current');
            li=li+1; h(li)=h4;
        end
        ylabel('Normalized Current'); ylim([-1 1]);

        % ---------- T1/T2 markers ----------
        yyaxis left;
        if isfield(waveData,'HV') && ~isnan(waveData.HV.T1), xline(waveData.HV.T1,':','Color',colH,'Alpha',0.6,'HandleVisibility','off'); end
        if isfield(waveData,'HV') && ~isnan(waveData.HV.T2), xline(waveData.HV.T2,'--','Color',colH,'Alpha',0.6,'HandleVisibility','off'); end
        if isfield(waveData,'FV') && ~isnan(waveData.FV.T1), xline(waveData.FV.T1,':','Color',colF,'Alpha',0.6,'HandleVisibility','off'); end
        if isfield(waveData,'FV') && ~isnan(waveData.FV.T2), xline(waveData.FV.T2,'--','Color',colF,'Alpha',0.6,'HandleVisibility','off'); end

        % ---------- Legend (only show what's plotted) ----------
        if li > 0
            legend(h(1:li), 'Location','southeast');
        end

    end
end

%% === Prepare Features and Labels ===
feature_cols = {'T1_voltage','T2_voltage','Vp_voltage','Ip_current','Beta_voltage'};
X = T{:, feature_cols};
Y = categorical(T.Label);
classes = categories(Y);
numClasses = numel(classes);
[X, mu, sigma] = zscore(X);
nSamples  = size(X,1);
nFeatures = size(X,2);
Xcnn = reshape(X', [nFeatures, 1, 1, nSamples]);

%% === Hold-Out Split ===
cv_holdout = cvpartition(Y, 'HoldOut', 0.2);
idxTr = training(cv_holdout);
idxTe = test(cv_holdout);

Xtrain = X(idxTr,:);   Ytrain = Y(idxTr);
Xtest  = X(idxTe,:);   Ytest  = Y(idxTe);
Xtrain_cnn = Xcnn(:,:,:,idxTr);
Xtest_cnn  = Xcnn(:,:,:,idxTe);
Ytrain_cnn = Ytrain;   Ytest_cnn = Ytest;

%% === Balance Training Data by Random Oversampling ===
[Xtrain_bal, Ytrain_bal] = oversample_minority(Xtrain, Ytrain);
Xtrain_bal_cnn = reshape(Xtrain_bal', [nFeatures, 1, 1, size(Xtrain_bal,1)]);

%% === Hyperparameter Tuning + Training - HOLDOUT ===

svmModel = fitcsvm(Xtrain_bal, Ytrain_bal, ...
    'KernelFunction','linear', ...
    'Standardize',true, ...
    'ClassNames',categorical({'Good','Fault'}), ...  % fix class order
    'OptimizeHyperparameters',{'BoxConstraint'}, ...
    'HyperparameterOptimizationOptions', struct( ...
        'UseParallel', false, 'ShowPlots', false, 'Verbose', 0));


% Random Forest (Bagged Trees)
rfModel = fitcensemble(Xtrain_bal, Ytrain_bal, ...
    'Method','Bag', 'Prior','uniform', 'Cost', [0 1; 5 0], ...
    'OptimizeHyperparameters',{'NumLearningCycles','MinLeafSize'}, ...
    'HyperparameterOptimizationOptions', struct('UseParallel',false,'ShowPlots',false,'Verbose',0));

% CNN
bestValAcc = 0; bestCNNModel = [];
learningRates = [1e-3, 5e-4];
dropouts      = [0.5, 0.3];

for lr = learningRates
    for dp = dropouts
        layersCNN = [
            imageInputLayer([nFeatures 1 1],'Normalization','none')
            convolution2dLayer([3 1],8,'Padding','same')
            batchNormalizationLayer
            reluLayer
            convolution2dLayer([3 1],16,'Padding','same')
            batchNormalizationLayer
            reluLayer
            fullyConnectedLayer(32)
            reluLayer
            dropoutLayer(0.5)
            fullyConnectedLayer(numClasses)
            softmaxLayer
            classificationLayer];



        optionsCNN = trainingOptions('adam', ...
            'InitialLearnRate', lr, 'MaxEpochs', 30, 'MiniBatchSize',16, ...
            'Shuffle','every-epoch','Verbose',false,'Plots','none', ...
            'ValidationData',{Xtest_cnn, Ytest_cnn}, 'ValidationFrequency',10, ...
            'ExecutionEnvironment','auto');

        net = trainNetwork(Xtrain_bal_cnn, Ytrain_bal, layersCNN, optionsCNN);
        YvalPred = classify(net, Xtest_cnn);
        valAcc = mean(YvalPred == Ytest_cnn);
        if valAcc > bestValAcc, bestValAcc = valAcc; bestCNNModel = net; end
    end
end
cnnModel = bestCNNModel;

%% === Evaluate on HoldOut test set ===
Ysvm    = predict(svmModel, Xtest);
Yrf     = predict(rfModel,  Xtest);
YcnnPred= classify(cnnModel, Xtest_cnn);

figure; confusionchart(Ytest, Ysvm, 'RowSummary','row-normalized','ColumnSummary','column-normalized'); title('SVM - HoldOut');
figure; confusionchart(Ytest, Yrf,  'RowSummary','row-normalized','ColumnSummary','column-normalized'); title('RF - HoldOut');
figure; confusionchart(Ytest_cnn, YcnnPred, 'RowSummary','row-normalized','ColumnSummary','column-normalized'); title('CNN - HoldOut');

% Metrics (force class order so TP=Fault as Fault)
models = {'SVM','RF','CNN'};
preds  = {Ysvm, Yrf, YcnnPred};
all_metrics = zeros(3,5); % Acc, Prec, Rec, F1, MCC
order = categorical({'Good','Fault'});

for m = 1:3
    conf = confusionmat(Ytest, preds{m}, 'Order', order);
    if size(conf,1)==2
        TP = conf(2,2); TN = conf(1,1); FP = conf(1,2); FN = conf(2,1);
    else
        TP=0; TN=0; FP=0; FN=0;
    end
    acc = sum(diag(conf))/sum(conf(:));
    prec = TP/(TP+FP+eps);
    rec  = TP/(TP+FN+eps);
    F1   = 2*(prec*rec)/(prec+rec+eps);
    MCC  = computeMCC(conf);
    all_metrics(m,:) = [acc,prec,rec,F1,MCC];
end

T_metrics = array2table(all_metrics, 'VariableNames',{'Accuracy','Precision','Recall','F1','MCC'}, 'RowNames',models);
disp(T_metrics);

%% === ROC Curves (robust; works with any subset of models) ===
disp('Plotting ROC Curves...');
figure; hold on; grid on;
title('ROC Curves - HoldOut Set'); xlabel('False Positive Rate'); ylabel('True Positive Rate');

leg = {};  % legend entries
Ypos = double(Ytest == 'Fault');  % positive-class indicator (unchanged)

% ---- Linear SVM (hold-out) ----
if exist('svmModel','var') && ~isempty(svmModel)
    [~, scores] = predict(svmModel, Xtest);
    cls = string(svmModel.ClassNames);            % order of score columns
    posIdx = find(cls == "Fault", 1);             % which column is 'Fault'?
    if ~isempty(posIdx) && size(scores,2) >= posIdx
        [fprSVM, tprSVM, ~, aucSVM] = perfcurve(Ypos, scores(:,posIdx), 1);
        plot(fprSVM, tprSVM, '-', 'LineWidth', 2);
        leg{end+1} = sprintf('Linear SVM (AUC=%.2f)', aucSVM);
    else
        warning('SVM: could not locate score column for ''Fault''.');
    end
end

% ---- Random Forest ----
if exist('rfModel','var') && ~isempty(rfModel)
    [~, scores_rf] = predict(rfModel, Xtest);
    cls = string(rfModel.ClassNames);
    posIdx = find(cls == "Fault", 1);
    if ~isempty(posIdx) && size(scores_rf,2) >= posIdx
        [fprRF, tprRF, ~, aucRF] = perfcurve(Ypos, scores_rf(:,posIdx), 1);
        plot(fprRF, tprRF, '--', 'LineWidth', 2);
        leg{end+1} = sprintf('Random Forest (AUC=%.2f)', aucRF);
    else
        warning('RF: could not locate score column for ''Fault''.');
    end
end

% ---- CNN ----
if exist('cnnModel','var') && ~isempty(cnnModel)
    scores_cnn = predict(cnnModel, Xtest_cnn);  % N x K probabilities
    % Class order for CNN follows the training labels:
    cls = string(categories(Ytrain_bal));
    posIdx = find(cls == "Fault", 1);
    if ~isempty(posIdx) && size(scores_cnn,2) >= posIdx
        [fprCNN, tprCNN, ~, aucCNN] = perfcurve(Ypos, scores_cnn(:,posIdx), 1);
        plot(fprCNN, tprCNN, ':', 'LineWidth', 2);
        leg{end+1} = sprintf('CNN (AUC=%.2f)', aucCNN);
    else
        warning('CNN: could not locate score column for ''Fault''.');
    end
end

if ~isempty(leg)
    legend(leg, 'Location','best');
else
    warning('No ROC curves were plotted (no models or no valid scores).');
end

%% === Bar Plot of Evaluation Metrics ===
disp('Plotting bar graph of evaluation metrics...');
figure;
bar(all_metrics', 'grouped');
xticklabels({'Accuracy','Precision','Recall','F1-score','MCC'});
legend(models, 'Location','northoutside','Orientation','horizontal');
ylabel('Score'); ylim([0 1]); grid on;
title('Model Performance Comparison (HoldOut)');

%% === Feature Importance ===
feature_names = {'T1_voltage','T2_voltage','Vp_voltage','Ip_current','Beta_voltage'};

% RF importance
rf_importance = predictorImportance(rfModel);
figure; bar(rf_importance, 'FaceColor',[0.2 0.6 0.5]);
xticks(1:numel(feature_names)); xticklabels(feature_names);
xlabel('Feature'); ylabel('Importance');
title('Feature Importance - Random Forest (HoldOut)');

% Linear SVM feature importance (use the trained hold-out model)
weights = abs(svmModel.Beta);
svm_importance = weights / sum(weights);
figure; bar(svm_importance, 'FaceColor',[0.5 0.4 0.7]);
xticks(1:numel(feature_names)); xticklabels(feature_names);
xlabel('Feature'); ylabel('Normalized Weight');
title('Feature Importance - Linear SVM (HoldOut)');

% CNN permutation importance
disp('Computing CNN Feature Importance (Permutation Style)...');
nFeat = size(Xtest,2); nSamples = size(Xtest,1);
base_preds = classify(cnnModel, Xtest_cnn);
acc_base = mean(base_preds == Ytest_cnn);
cnn_importance = zeros(1,nFeat);
for ff = 1:nFeat
    Xp = Xtest;
    Xp(:,ff) = Xp(randperm(nSamples), ff);
    Xp_cnn = reshape(Xp', [nFeat, 1, 1, nSamples]);
    preds_perm = classify(cnnModel, Xp_cnn);
    acc_perm = mean(preds_perm == Ytest_cnn);
    cnn_importance(ff) = acc_base - acc_perm;
end
cnn_importance = cnn_importance / (sum(cnn_importance)+eps);
figure; bar(cnn_importance, 'FaceColor',[0.8 0.4 0.2]);
xticks(1:nFeat); xticklabels(feature_names);
xlabel('Feature'); ylabel('Normalized Importance');
title('CNN Feature Importance (Permutation-Based)');

%% === 5-Fold Cross-Validation with Oversampling ===
k = 5;
cv = cvpartition(Y,'KFold',k);
metrics_cv = zeros(3,5);  % Acc, Prec, Rec, F1, MCC
models = {'SVM','RF','CNN'};

for i = 1:k
    idxTr = training(cv,i);
    idxTe = test(cv,i);

    Xtr = X(idxTr,:);
    Ytr = Y(idxTr);
    Xte = X(idxTe,:);
    Yte = Y(idxTe);

    % Oversample training for SVM/RF; test remains original
    [Xtr_bal, Ytr_bal] = oversample_minority(Xtr, Ytr);

    % CNN shapes
    nTe = size(Xte,1);
    Xte_cnn = reshape(Xte', [nFeatures, 1, 1, nTe]);
    nTr = size(Xtr_bal,1);
    Xtr_bal_cnn = reshape(Xtr_bal', [nFeatures, 1, 1, nTr]);

    % Train simple models for CV speed
    svmModel_cv = fitcsvm(Xtr_bal, Ytr_bal, 'KernelFunction','linear', 'Standardize',true);
    rfModel_cv  = fitcensemble(Xtr_bal, Ytr_bal, 'Method','Bag', 'NumLearningCycles',30);

    % ---- CNN (note: no undefined 'dp'; fixed dropout 0.5) ----
    layersCNN = [
        imageInputLayer([nFeatures 1 1],'Normalization','none')
        convolution2dLayer([3 1],8,'Padding','same')
        batchNormalizationLayer
        reluLayer
        convolution2dLayer([3 1],16,'Padding','same')
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(32)
        reluLayer
        dropoutLayer(0.5)
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];

    optionsCNN = trainingOptions('adam', ...
        'InitialLearnRate', 1e-3, ...
        'MaxEpochs', 20, ...
        'MiniBatchSize', 16, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'Plots', 'none', ...
        'ExecutionEnvironment', 'auto');

    net_cv = trainNetwork(Xtr_bal_cnn, Ytr_bal, layersCNN, optionsCNN);

    % Predict on test fold
    Ysvm_cv = predict(svmModel_cv, Xte);
    Yrf_cv  = predict(rfModel_cv,  Xte);
    Ycnn_cv = classify(net_cv, Xte_cnn);

    % Score this fold
    preds_cv = {Ysvm_cv, Yrf_cv, Ycnn_cv};
    order = categorical({'Good','Fault'});  % fix class order

    for m = 1:3
        conf = confusionmat(Yte, preds_cv{m}, 'Order', order);
        TP = conf(2,2); TN = conf(1,1); FP = conf(1,2); FN = conf(2,1);
        acc = sum(diag(conf))/sum(conf(:));
        prec = TP/(TP+FP+eps);
        rec  = TP/(TP+FN+eps);
        F1   = 2*(prec*rec)/(prec+rec+eps);
        MCC  = computeMCC(conf);
        metrics_cv(m,:) = metrics_cv(m,:) + [acc,prec,rec,F1,MCC];
    end
end

metrics_cv = metrics_cv / k; % average across folds
T_cv_metrics = array2table(metrics_cv, ...
    'VariableNames', {'Accuracy','Precision','Recall','F1','MCC'}, ...
    'RowNames', models);
disp('5-Fold Cross-Validation Average Metrics:');
disp(T_cv_metrics);

figure;
bar(metrics_cv', 'grouped');
xticklabels({'Accuracy','Precision','Recall','F1-score','MCC'});
legend(models, 'Location', 'northoutside', 'Orientation', 'horizontal');
ylabel('Score'); ylim([0 1]); grid on;
title('Model Performance Comparison (5-Fold CV)');


%% ======================= Local helper functions =======================

function arr = toNumericArray(inputVar)
if iscell(inputVar)
    arr = cellfun(@str2double, inputVar);
elseif isstring(inputVar) || ischar(inputVar)
    arr = str2double(inputVar);
else
    arr = double(inputVar);
end
end

function feats = extract_waveform_features(t, v, i)
% Returns {T1, T2, Vp, Ip, Beta}
t = t(:); v = v(:); i = i(:);
t = fillmissing(t,'linear','EndValues','nearest');
v = fillmissing(v,'linear','EndValues','nearest');
i = fillmissing(i,'linear','EndValues','nearest');

% Normalize for front timing (0..1)
v_shift = v - min(v);
Vp      = max(v_shift);
v_norm  = v_shift / (Vp + eps);

% Front time T1: 10%->90% or 30%->90% (IEC variants exist)
idx10 = find(v_norm >= 0.1, 1, 'first');
idx90 = find(v_norm >= 0.9, 1, 'first');
if ~isempty(idx10) && ~isempty(idx90)
    T1 = t(idx90) - t(idx10);
else
    T1 = NaN;
end

% Tail time T2: time from peak to 50% of Vp on the tail
[~, peakIdx] = max(v_shift);
tailIdx = find(v_shift(peakIdx:end) <= 0.5*Vp, 1, 'first');
if ~isempty(tailIdx)
    T2 = t(peakIdx + tailIdx - 1) - t(peakIdx);
else
    T2 = NaN;
end

% Peak current
Ip = max(abs(i));

% Beta = T1/T2
if ~isnan(T1) && ~isnan(T2) && T2 ~= 0
    Beta = T1 / T2;
else
    Beta = NaN;
end

feats = {T1, T2, Vp, Ip, Beta};
end

function [X_bal, Y_bal] = oversample_minority(X, Y)
% Random oversampling of minority classes
classes = categories(Y);
counts  = countcats(Y);
maxCount = max(counts);
nFeatures = size(X,2);

totalToAdd = sum(maxCount - counts);
X_add = zeros(totalToAdd, nFeatures);
Y_add = categorical(strings(totalToAdd,1), classes);

k = 0;
for c = 1:numel(classes)
    cls = classes{c};
    idx = find(Y==cls);
    nToAdd = maxCount - numel(idx);
    if nToAdd > 0
        pick = randsample(idx, nToAdd, true);
        X_add(k+(1:nToAdd),:) = X(pick,:);
        Y_add(k+(1:nToAdd))  = Y(pick);
        k = k + nToAdd;
    end
end

X_bal = [X; X_add];
Y_bal = [Y; Y_add];
end

function mcc = computeMCC(conf)
if size(conf,1)~=2, mcc = NaN; return; end
TP = conf(2,2); TN = conf(1,1); FP = conf(1,2); FN = conf(2,1);
numerator = (TP*TN) - (FP*FN);
denominator = sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
if denominator==0, mcc = 0; else, mcc = numerator/denominator; end
end

function xN = normalize_signal(x)
% Remove pre-impulse baseline (first 2%), then scale by max |.| safely
x = x(:);
if isempty(x), xN = x; return; end
n0 = max(1, round(0.02*numel(x)));
base = mean(x(1:n0), 'omitnan');
x0 = x - base;
s  = max(abs(x0), [], 'omitnan');
if isempty(s) || isnan(s) || s==0, s = 1; end
xN = x0 / s;
end

function shaded_mean(x, mu, sd, clr, fa)
% Draw shaded mean ± sd behind the line (hidden from legend)
fill([x(:); flipud(x(:))], [mu(:)+sd(:); flipud(mu(:)-sd(:))], ...
    clr, 'EdgeColor','none','FaceAlpha',fa, 'HandleVisibility','off');
end

