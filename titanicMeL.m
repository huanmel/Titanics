%%% Kaggle Пример Titanic
%% startup
addpath(pwd);
addpath([pwd '\files']);
%% Импорт
load data
 %tr=readtable('train.csv','Delimiter',',','format','%d%d%d%q%s%u8%u8%u8%s%n%s%s');
 %PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
 %tr.Properties.VariableNames
 %tt=readtable('test.csv','Delimiter',',','format','%d%d%q%s%u8%u8%u8%s%n%s%s'); 
  %PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
  
%% Категориальные переменные
tr.Survived=logical(tr.Survived);
tr.Sex=categorical(tr.Sex,'Ordinal',false);
tr.Embarked=categorical(tr.Embarked);
tt.Sex=categorical(tt.Sex,'Ordinal',false);
tt.Embarked=categorical(tt.Embarked);
% Добавим переменную в тестовый набор
tt.Survived=logical(zeros(size(tt.Sex)));
%%
tt.S=tt.Sex=='female';
tr.S=tr.Sex=='female';
%% Выборки данных
cv = cvpartition(height(tr),'holdout',0.45);
Y=tr.Survived;
X=tr;
X.Survived=[];
X.Name=[];
X.Sex=[];
%% Training set

Xtrain = X(training(cv),:);
Ytrain = Y(training(cv),:);
% Test set
Xtest = X(test(cv),:);
Ytest = Y(test(cv),:);
% 
XNum=X;
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
%%
subplot(1,2,1), scatter3(tr.Survived,tr.Pclass,tr.Age,[],tr.S); xlabel('Survived'); ylabel('Pclass'); zlabel('Age'); 
subplot(1,2,2), scatter3(tr.Survived,tr.Pclass,tr.Fare,[],tr.S); xlabel('Survived'); ylabel('Pclass'); zlabel('Fare'); 
title('Color = sex 1=female')
summary([tr.Sex categorical(tr.Survived)])% Сколько выжило/не выжило
figure
boxplot(tr.Fare,tr.Survived)
%%
cols=[3,6,7,8,10,12,13];
for i = cols
    figure
    boxplot(tr{:,i},tr.Survived)
end
%%
grpstats(tr, {'Sex'},{'@(x)sum(x,1)/numel(x)','@(x)sum(x,1)', 'numel'},'DataVars',{'Survived'})
% видно, что можно прогнозировать по гендерному признаку, попробуем дальше

%%
% Делаем прогноз на тестовом множестве
%Y_g = Xtest.Sex=='female';
Y_g = Xtest.S==1;

% Вычисление матрицы несоответствия
C_g = confusionmat(Ytest,Y_g)
C_g=dispConfusion(C_g, 'По половому признаку',{'Нет', 'Да'});
S_g= cm(Ytest,Y_g)
%%
grpstats(tr, {'Survived', 'Pclass','Sex'},{'@(x)sum(x)/numel(x)'},'DataVars',{'S'} )
%%
[net] = NNfun2(double(XtrainNum),YtrainNum*2-1);
% Делаем прогноз на тестовом множестве
Yscore_nn = net(XtestNum')';
%%
[Y_nn,S_n,pmax]=findpor(Yscore_nn,YtestNum);
%Y_nn = round(Yscore_nn);
%%
da = fitcdiscr(XtrainNum,Ytrain);
% Делаем прогноз по тестовому множесту
[Y_da, Yscore_da] = predict(da,XtestNum); 
Yscore_da = Yscore_da(:,2);
 S_d= cm(YtestNum,Y_da)
 %%
t = fitctree(XtrainNum,Ytrain);
Y_t = predict(t,XtestNum);
 S_t= cm(YtestNum,Y_t)
 %%
 opts = statset('UseParallel',true);
retrain = false;
if retrain || ~exist('models.mat','file')
   % Обучение классификатора
    tb = TreeBagger(10,XtrainNum,Ytrain,'method','classification',...
        'Options',opts,'OOBVarImp','on');
    save BaggedTreeSmall.mat tb
else
    load BaggedTreeSmall.mat tb
end
    %%
    % Делаем прогноз по тестовому множесту
[Y_tb, Yscore_tb] = predict(tb,XtestNum);
S_tb= cm(YtestNum,double(Y_tb))