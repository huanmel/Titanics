function [y]=cm(y1,y2);
% �������� - ��������� ����������� 
% ������: S_tb= cm(YtestNum,double(Y_tb)), 
% ��� S_tb - �������� �������� ��� �������������� tb
% Y_tb - ����� �������������� tb
% YtestNum - ��������� ��������

y=sum(y1==y2)/numel(y1);
end