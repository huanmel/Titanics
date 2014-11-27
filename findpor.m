function [Y_nn,S_n,pmax]=findpor(Yscore_nn,YtestNum)
% Функция поиска оптимального значения порога
p1=min(Yscore_nn);
p2=max(Yscore_nn);
pi=linspace(p1,p2,25);
pmx=zeros(numel(pi),2);
pmx(:,1)=pi;
for i=1:numel(pi)
    Y_nn=Yscore_nn>=pi(i);
   % S_n= cm((YtestNum+1)/2,Y_nn);
    S_n= cm(YtestNum,Y_nn);
    pmx(i,2)=S_n;
    disp(pmx(i,:))
    
end;
plot(pmx(:,1),pmx(:,2))
[i1,i2]=max(pmx);
disp('Максимум')
disp(pmx(i2(2),:))
S_n=pmx(i2(2),2);
pmax=pmx(i2(2),1);
Y_nn=Yscore_nn>=pmax;