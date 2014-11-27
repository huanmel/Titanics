%%% ������ 
% ����� ����������� ����� ��� ������
%% startup
addpath(pwd);
addpath([pwd '\files']);
%% ������
load data
%% �������������� ����������
tr.Survived=logical(tr.Survived);
tr.Sex=categorical(tr.Sex,'Ordinal',false);
tr.Embarked=categorical(tr.Embarked);
tt.Sex=categorical(tt.Sex,'Ordinal',false);
tt.Embarked=categorical(tt.Embarked);
% ������� ���������� � �������� �����
tt.Survived=logical(zeros(size(tt.Sex)));
%% Sex=0|1
tt.S=tt.Sex=='female';
tr.S=tr.Sex=='female';
%% ������� ������, ���������� ����
cv = cvpartition(height(tr),'holdout',0.40);
Y=tr.Survived;
X=tr;
X.Survived=[];
X.Name=[];
X.Sex=[];

%% Training set/Test set

Xtrain = X(training(cv),:);
Ytrain = Y(training(cv),:);
% Test set
Xtest = X(test(cv),:);
Ytest = Y(test(cv),:);
% 
XNum=X;
%���������� �������� ����������
XNum.Ticket=[];
XNum.Cabin=[]; %TODO: ����������� ����� �� ��������� ������������� ������������ ������ ���������

Xde=  dummyvar(X.Embarked);
XNum.EC=Xde(:,1);
XNum.EQ=Xde(:,2);
XNum.ES=Xde(:,3);
XNum.Embarked=[];
names = XNum.Properties.VariableNames;
XtrainNum = double(table2array(XNum(training(cv),:)));
YtrainNum = double(Ytrain)%*2-1;
XtestNum = double(table2array(XNum(test(cv),:)));
YtestNum = double(Ytest)%*2-1;

disp('Training Set')
tabulate(Ytrain)
disp('Test Set')
tabulate(Ytest)

%% ������ � �������
% ���� ����� ���������� ������, ����� ���� �������

%% ������ � �����������
% ������� ���� ���������

%% ������ ����� ���������
% S_tb= cm(YtestNum,double(Y_tb)),

%% ��������� ����������
% ����� ������������ ��� ��������� ����������
