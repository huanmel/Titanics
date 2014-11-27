%%% Каркас 
% здесь стандартные блоки для работы
%% startup
addpath(pwd);
addpath([pwd '\files']);
%% Импорт
load data
%% Категориальные переменные
tr.Survived=logical(tr.Survived);
tr.Sex=categorical(tr.Sex,'Ordinal',false);
tr.Embarked=categorical(tr.Embarked);
tt.Sex=categorical(tt.Sex,'Ordinal',false);
tt.Embarked=categorical(tt.Embarked);
% Добавим переменную в тестовый набор
tt.Survived=logical(zeros(size(tt.Sex)));
%% Sex=0|1
tt.S=tt.Sex=='female';
tr.S=tr.Sex=='female';
%% Выборка данных, отложенный тест
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
%Выкидываем ненужные переменные
XNum.Ticket=[];
XNum.Cabin=[]; %TODO: Попробовать потом на основании классификации расположения кабины поставить

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

%% Работа с данными
% Ищем какие переменные важные, может быть графики

%% Работа с алгоритмами
% Обучаем свои алгоритмы

%% Пример теста алгоритма
% S_tb= cm(YtestNum,double(Y_tb)),

%% Сравнение алгоритмов
% здесь наборфункций для сравнения алгоритмов
