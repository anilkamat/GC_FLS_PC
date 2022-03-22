clc; clear all; close all;
addpath 'D:\\RPI\\ResearchWork\\Papers_\\Effective_Connectivity\\toolbox_original\\mvgc_v1.0\\Results_TRGC_Nov_Exp_Nature'
PhySimu = load('TRGC_Exp_Nov_PhysicalSimulator.mat'); % import data of physical simulator
VirSimu = load('TRGC_Exp_Nov_VirtualSimulator.mat');    % import data of virtual simulator
fpath2 = 'D:\RPI\ResearchWork\Papers_\Effective_Connectivity\toolbox_original\mvgc_v1.0\Results_TRGC_Nov_Exp_Nature\P_value_corrected_FDR';

%% ################ H1: pair wise anova test between Physical and virtual simulator, head to head comparision ########################
%######################## within expert and within novice
X = [];
p1 = [];
pVal = [];
baseFileName = 'H1a_Phy_Vir_Simulator.xlsx';          % excel write
fullFileName = fullfile(fpath2, baseFileName);
for p = 1:2     % 1--> exp, 2 --> nov
    p_Val = [];
    if p ==1
        cohort = sprintf('B_Exp');
        temp = sprintf('Exp');
    else
        cohort = sprintf('B_Nov');
        temp = sprintf('Nov');
    end
    nnn = min(size(fieldnames(PhySimu.(cohort)),1), size(fieldnames(VirSimu.(cohort)),1));   % common number of trails between physical and virtual simulator
    for k = 1:nnn  % nnn loop over trial
        col = 1;
        trial_temp = sprintf('Tr_%d',k);
        %     if k > 4
        %        n = 5; % after five trail, no.of subj in VBLAST drops to 5
        %     end
        nn = min(size(PhySimu.(cohort).(trial_temp).(temp),3), size(VirSimu.(cohort).(trial_temp).(temp),3));
        for j = 1:nn-1    % nn window no.
            Win = sprintf('Win_%d',j);
            n = min(size(PhySimu.(cohort).(trial_temp).(temp),1), size(VirSimu.(cohort).(trial_temp).(temp),1));
            for i = 1:20    % 20 connection
                close all;
                X = [];
                if p == 1
                    X = [PhySimu.(cohort).(trial_temp).Exp(1:n,i,j), VirSimu.(cohort).(trial_temp).Exp(1:n,i,j)];
                    tbl = ['tbl',num2str(i)];
                    N = size(X,2);
                    kkk = 1;
                    for iii = 1:N % to store the statistics over each trial
                        p_Val.(trial_temp)(i,iii+kkk) =  nanmean(X(:,iii)); % Mean mean of first column i.e. connection ij
                        p_Val.(trial_temp)(i,iii+kkk+1) =  std(X(:,iii)); % std of first column i.e. connection ij
                        kkk = kkk+1;
                    end
                    [p_Val.(trial_temp)(i,2*kkk+1),tbl,~] = anova1(X);    % pval -> connection x windows
                    p_Val.(trial_temp)(i,2*kkk) =  tbl{2,5}; % F-Statistics
                    [pVal.(cohort).Exp.(trial_temp)(i,1,j),tbl,~] = anova1(X);
                    sheet = sprintf('Exp_T%d%s',k,Win)
                else
                    X = [PhySimu.(cohort).(trial_temp).Nov(1:n,i,j), VirSimu.(cohort).(trial_temp).Nov(1:n,i,j)];
                    tbl = ['tbl',num2str(i)];
                    N = size(X,2);
                    kkk = 1;
                    for iii = 1:N % to store the statistics over each trial
                        p_Val.(trial_temp)(i,iii+kkk) =  nanmean(X(:,iii)); % Mean mean of first column i.e. connection ij
                        p_Val.(trial_temp)(i,iii+kkk+1) =  std(X(:,iii)); % std of first column i.e. connection ij
                        kkk = kkk+1;
                    end
                    [p_Val.(trial_temp)(i,2*kkk+1),tbl,~] = anova1(X);    % pval -> connection x windows
                    p_Val.(trial_temp)(i,2*kkk) =  tbl{2,5}; % F-Statistics
                    [pVal.(cohort).Nov.(trial_temp)(i,1,j),tbl,~] = anova1(X);
                    sheet = sprintf('Nov_T%d%s',k,Win);
                end
            end
            %fnPlot(p1.(trial_temp)(:,1,j), trial_temp,j,Cnames,fpath2);
            %xlswrite(fullFileName,p_Val.(trial_temp),sheet);
        end
    end
end
close all;
%% ################ H1: pair wise anova test between Physical and virtual simulator, head to head comparision ########################
%######################## Expert and Novice put together
group = {'FLS';'VBLaST'};
X = [];
pVal_Nov_Exp_tog = [];
baseFileName = 'H6_Phy_Vir_Simulator_Exp_NovTogether.xlsx';          % excel write
fullFileName = fullfile(fpath2, baseFileName);
sheet = 'Exp_T1';
col_name = 'A':'Z';
for p = 1:1
    temp = [];
    if p ==1
        cohort = sprintf('B_Exp');
        temp = sprintf('Exp');
    else
        cohort = sprintf('B_Nov');
        temp = sprintf('Nov');
    end
    nnn = min(size(fieldnames(PhySimu.(cohort)),1), size(fieldnames(VirSimu.(cohort)),1));   % common number of trails between physical and virtual simulator
    for k = 1:nnn  % nnn loop over trial
        col = 1;
        trial_temp = sprintf('Tr_%d',k);
        %     if k > 4
        %        n = 5; % after five trail, no.of subj in VBLAST drops to 5
        %     end
        nn = min([size(PhySimu.B_Exp.(trial_temp).Exp,3), size(VirSimu.B_Exp.(trial_temp).Exp,3),...
            size(PhySimu.B_Nov.(trial_temp).Nov,3), size(VirSimu.B_Nov.(trial_temp).Nov,3)]);
        for j = 1:nn    % nn window no.
            n = min([size(PhySimu.B_Exp.(trial_temp).Exp,1), size(VirSimu.B_Exp.(trial_temp).Exp,1),...
                size(PhySimu.B_Nov.(trial_temp).Nov,1), size(VirSimu.B_Nov.(trial_temp).Nov,1)]);
            close all;
            for i = 1:20    % 20 connection
                X = [];
                if p == 1
                    X = [[PhySimu.('B_Exp').(trial_temp).Exp(1:n,i,j) ;PhySimu.('B_Nov').(trial_temp).Nov(1:n,i,j)],[ VirSimu.('B_Exp').(trial_temp).Exp(1:n,i,j) ; VirSimu.('B_Nov').(trial_temp).Nov(1:n,i,j)]];
                    tbl = ['tbl',num2str(i)];
                    [pVal_Nov_Exp_tog.(cohort).Exp.(trial_temp)(i,1,j),tbl,stats] = anova1(X);    % Phy vs virtual when novice and expert are put together in one group
                else
                    X = [PhySimu.(cohort).(trial_temp).Nov(1:n,i,j), VirSimu.(cohort).(trial_temp).Nov(1:n,i,j)];
                    tbl = ['tbl',num2str(i)];
                    [pVal_Nov_Exp_tog.(cohort).Nov.(trial_temp)(i,1,j),tbl,stats] = anova1(X);
                end
                %figure(10)
                %baseFileName = sprintf('FLS vs VBLaST Aonva Conn:-%s.png',connection_GC{i});
                
                %             title(baseFileName)
                %             xticklabels(group);
                %             xtickangle(45)
                %             fullFileName = fullfile(fpath, baseFileName);
                %saveas(figure(10),fullFileName)
            end
            %fnPlot(p1.(trial_temp)(:,1,j), trial_temp,j,Cnames,fpath2);
            if p ==1
                sheet = sprintf('Exp_T%d',k);
                %xlswrite(fullFileName,pVal_Nov_Exp_tog.(cohort).Exp.(trial_temp)(:,1,j),sheet,col_name(col+2));
                col = col+1;
            else
                sheet = sprintf('Nov_T%d',k);
                %xlswrite(fullFileName,pVal.(cohort).Nov.(trial_temp)(:,1,j),sheet,col_name(col+2));
                col = col+1;
            end
            
        end
    end
end
%% find the index of pvalue < 0.01
for i =1:2
    if i == 1
        t = size(fieldnames(pVal.('B_Exp').Exp),1);
        %temp = sprintf('pVal.('B_Nov').Nov');
    else
        t = size(fieldnames(pVal.('B_Nov').Nov),1);
    end
    for j = 1:t
        tr = sprintf('Tr_%d',j);
        if i ==1
            temp = pVal.B_Exp.Exp.(tr);
            [row,colm,wid ]=  find(temp < 0.01);
        else
            temp = pVal.B_Nov.Nov.(tr);
            [row,colm,wid ]=  find(temp < 0.01);
        end
    end
end
