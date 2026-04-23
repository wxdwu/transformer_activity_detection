function [recAngleMatrix, EST_SELECTED_PILOT_SET] = AMP_LiuLiang(pilotMatrix,rxAngleMatrix,txAngleMatrix,...
    angleMatrix,maxItr,SELECTED_PILOT_SET,NUM_ALL_PILOT,NUM_PILOT_ACROSS_SUBCARRIER,SNR)

% angleMatrix2 = reshape(angleMatrix,size(angleMatrix,1),size(angleMatrix,2)*size(angleMatrix,3),1);
% rxAngleMatrix2 = reshape(rxAngleMatrix,size(rxAngleMatrix,1),size(rxAngleMatrix,2)*size(rxAngleMatrix,3),1);
% txAngleMatrix2 = reshape(txAngleMatrix,size(txAngleMatrix,1),size(txAngleMatrix,2)*size(txAngleMatrix,3),1);
% PL = std(angleMatrix2,0,2).^2;
% PL(setdiff([1:NUM_ALL_PILOT],SELECTED_PILOT_SET)) = sum(PL(SELECTED_PILOT_SET))/length(SELECTED_PILOT_SET);
% sigma_w = std2(rxAngleMatrix2 - txAngleMatrix2);
% [~,recAngleMatrix2,~,tau_real,tau_est] = noisyCAMPmmseforKLS(pilotMatrix,rxAngleMatrix2,angleMatrix2,maxItr,size(SELECTED_PILOT_SET)/NUM_ALL_PILOT,PL,sigma_w);
% recAngleMatrix = reshape(recAngleMatrix2,size(angleMatrix,1),size(angleMatrix,2),size(angleMatrix,3));

 
for itr1 = 1 : NUM_PILOT_ACROSS_SUBCARRIER
            PL = std(angleMatrix(:, :, itr1),0,2).^2; 
            %PL(setdiff([1:NUM_ALL_PILOT],SELECTED_PILOT_SET)) = min(PL(SELECTED_PILOT_SET));
            %PL(setdiff([1:NUM_ALL_PILOT],SELECTED_PILOT_SET)) = (min(PL(SELECTED_PILOT_SET)) + max(PL(SELECTED_PILOT_SET))) / 2;
            %PL(setdiff([1:NUM_ALL_PILOT],SELECTED_PILOT_SET)) = sum(PL(SELECTED_PILOT_SET)) / length(SELECTED_PILOT_SET);
            %PL(setdiff([1:NUM_ALL_PILOT],SELECTED_PILOT_SET)) = median(PL(SELECTED_PILOT_SET)) ;
            %PL(setdiff([1:NUM_ALL_PILOT],SELECTED_PILOT_SET)) = 0.01*min(PL(SELECTED_PILOT_SET));
            PL(setdiff([1:NUM_ALL_PILOT],SELECTED_PILOT_SET)) = 0;
            %sigma_w = std2(rxAngleMatrix(:, :, itr1) - txAngleMatrix(:, :, itr1));
            sigma_w = std2(rxAngleMatrix(:, :, itr1))*10^(-SNR/20);
            [~,recAngleMatrix(:, :, itr1),~,tau_real,tau_est] = noisyCAMPmmseforKLS(pilotMatrix,rxAngleMatrix(:, :, itr1),angleMatrix(:, :, itr1),maxItr,size(SELECTED_PILOT_SET)/NUM_ALL_PILOT,PL,sigma_w);
end 
 
SUM_POWER = sum(sum(abs(recAngleMatrix),2),3);
[~, i_sort] = sort(SUM_POWER,'descend');

EST_SELECTED_PILOT_SET = i_sort(1:length(SELECTED_PILOT_SET))';
EST_SELECTED_PILOT_SET = sort(EST_SELECTED_PILOT_SET);



















