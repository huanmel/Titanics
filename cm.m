function [y]=cm(y1,y2);
% Критерий - сравнение результатов 
% Пример: S_tb= cm(YtestNum,double(Y_tb)), 
% где S_tb - значение качества для классификатора tb
% Y_tb - выход классификатора tb
% YtestNum - известные значения

y=sum(y1==y2)/numel(y1);
end