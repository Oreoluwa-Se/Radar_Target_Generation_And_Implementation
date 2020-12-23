clear; clc;
% Givens
c = 3*10^8;

% Radar Specifications
Frequency = 77e9; %HZ
Range_res = 1; %m
Max_Range = 200; %m
Max_Vel = 70; %m/s
velocity_res = 3; %m/s
Nd=128; % number of chirps in one sequence
Nr=1024; % number of samples

% initial velociry and position
v_init = -50; %m/s
pos_init = 30; %m

%% Design the FMCW waveform
% Calculating bandwith
B_sweep = c/(2*Range_res);

% calculating chirp time
T_chirp = 5.5*2*Max_Range/c;

% calculating chirp slope
alpha = B_sweep/T_chirp;

%% Signal Propagation Modeling and beat signal creation
% Time samples
t = linspace(0, Nd*T_chirp, Nr*Nd);

% create vectors Transmit and Recieve signal
Tx = zeros(1, length(t)); % transmitted signal
Rx = zeros(1, length(t)); % recieved signal

% create range and time dealy signal
r_ = zeros(1, length(t)); % transmitted signal
td = zeros(1, length(t)); % recieved signal

% Running the radar scenario over the time
for i=1:length(t)
    fc = Frequency;
    
    % update range of the target for constant velocity
    r_(i) = pos_init + v_init*t(i);
    td(i) = 2*r_(i)/c;
    
    % calculate the transmitted and recieved signal
    Tx(i) = cos(2*pi*(fc*t(i) + alpha*(t(i)^2)/2));
    Rx(i) = cos(2*pi*(fc*(t(i)-td(i)) + 0.5*alpha*(t(i)-td(i))^2));
end

% create vectors for beat signal 
beat = Tx.*Rx;

%% FFT Operation
% reshape vector into Nr*Nd
beat = reshape(beat, [Nr, Nd]);

% run the FFT on beat signal along range bins, take absolute value
fft_1D = abs(fft(beat, Nr));

% normalize
fft_1D = fft_1D/max(fft_1D); 

% keep one half of the result
fft_1D = fft_1D(1:Nr/2 +1);

% plot fft1D
figure ('Name','Range from First FFT')
plot(fft_1D)
axis ([0 Max_Range 0 1]);
ylabel('Normalized Amplitude')
xlabel('Range');

%% 2dFF : Range Doppler Map
% 2D FFT using the FFT size for both dimensions.
sig_fft2 = fft2(beat,Nr,Nd);

% Taking just one side of signal from Range dimension.
sig_fft2 = sig_fft2(1:Nr/2,1:Nd);
sig_fft2 = fftshift (sig_fft2);
RDM = abs(sig_fft2);
RDM = 10*log10(RDM) ;

%use the surf function to plot the output of 2DFFT and to show axis in both
%dimensions
doppler_axis = linspace(-100,100,Nd);
range_axis = linspace(-Max_Range,Max_Range,Nr/2)*((Nr/2)/(2*Max_Range));

figure ('Name','surface plot of FFT2')
surf(doppler_axis,range_axis,RDM);
title('FFT2 surface plot')
xlabel('speed')
ylabel('Range')
zlabel('Amplitude')

%% 2D CFAR on RDM map
% Number of training and guard cells in both dimensions
Tcr = 20;
Tcd = 8;

Gcr = 12;
Gcd = 6;

% offset value
offset = 1.5;

% range and doppler sizes
train_range_size = (2*Tcr +2*Gcr + 1);
train_doppler_size = (2*Tcd +2*Gcd + 1);

% vector to store grid size
gridSize = train_range_size*train_doppler_size;
numTrainCell = gridSize - (2*Gcr + 1)*(2*Gcd + 1);

% storage for cell under test
CUT = zeros(size(RDM));

% loop through range and doppler
for i = 1:(size(RDM,1) - train_range_size)
    for j = 1:(size(RDM,2) - train_doppler_size)
        % obtain the index for current cell under test
        r_index = i + Tcr + Gcr;
        d_index = j + Tcd + Gcd;
        
        % for current patch, convert log to linear
        trainPatch = db2pow(RDM(i:i+train_range_size,j:j+train_doppler_size));
        
        % make none training cell regions = zero
        trainPatch(Tcr+1:end-Tcr, Tcd+1:end-Tcd) = 0;
        
        % calculate average noise level multiplying by offset
        noise_lvl = offset * pow2db(sum(sum(trainPatch))/numTrainCell);
        
        % check if RDM value is greater than signal
        if RDM(r_index, d_index) > noise_lvl
            CUT(r_index, d_index) = 1;
        end                 
    end
end

%% output
figure('Name', 'CA-CFAR Filtered RDM')
surf(doppler_axis,range_axis,CUT);
colorbar;
title( 'CA-CFAR Filtered RDM surface plot');
xlabel('Speed');
ylabel('Range');
zlabel('Normalized Amplitude');

figure('Name', 'CA-CFAR Filtered Range')
surf(doppler_axis,range_axis,CUT);
colorbar;
title( 'CA-CFAR Filtered RDM surface plot');
xlabel('Speed');
ylabel('Range');
zlabel('Normalized Amplitude');
view(90,0);

figure('Name', 'CA-CFAR Filtered Speed')
surf(doppler_axis,range_axis,CUT);
colorbar;
title( 'CA-CFAR Filtered RDM surface plot');
xlabel('Speed');
ylabel('Range');
zlabel('Normalized Amplitude');
view(0,0);








