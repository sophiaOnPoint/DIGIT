
% load data. Experimental data is taken when light is off, integrate at
% different integration time.
load('/run/user/1001/gvfs/afp-volume:host=houston.mit.edu,volume=qpgroup/Experiments/MontanaII/2023_0421_background/QE/Experiments_WideFieldParameterSweep2023_08_11_00_37_42.mat')
% extract captured images
im = data.data.data.images;
%%
% X,Y are the number of size pixels of the image
X = size(im,1);
Y = size(im,2);

% expNum is the size of the sweep number of exposure time
expNum = size(im,3);

% numIm is the number of image captured at each exposure time
numIm = size(im,4);

% flatten the image
imRe = reshape(double(im),[X*Y,expNum,numIm]);
%%
% Nonuniformity = standard deviation / mean(pixel value)
% Calculate the standard deviation of camera pixel value and reshape
imDev = reshape(std(imRe,1),[expNum,numIm]);

% Calculate the mean of camera pixel value and minus the camera offset.
% Define camera offset value;
bias = 1273;
imMean = reshape(mean(imRe),[expNum,numIm])-1273;
nonUni = imDev./imMean;
%%
% calculate the shot noise and its spatial distribution
% define camera settings: emgain = 3, quantum efficiency = 0.6;
gain = 3; qe = 0.6;

% shot noise is the square root of mean photon counts
simDev = sqrt(mean(imMean,2)*gain*qe);

% shot noise distribution
simUni = simDev./mean(imMean,2);
%%
subplot(2,1,1)
errorbar(data.data.data.exposureTime(1:6),mean(nonUni,2)*100,std(nonUni,[],2)*100/2,'o-','Color', colorGreen, 'MarkerSize',10,'LineWidth',1.5)
hold on
plot(data.data.data.exposureTime(1:6),simUni*100,'.-','Color', colorPurpleLight, 'MarkerSize',20,'LineWidth',1.5);
ylabel('Deviation (%)','FontSize',15,'FontName','Times New Roman')
xlabel('Integration Time (ms)','FontSize',15,'FontName','Times New Roman')
set(gca,'FontSize',15,'FontName','Times New Roman')
% Set x-axis and y-axis to log scale
set(gca, 'XScale', 'log', 'YScale', 'log');