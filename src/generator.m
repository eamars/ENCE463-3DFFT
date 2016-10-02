% Ran Bao
% University of Canterbury
clc;
clear;
close all;

% generate a 3d array
A = rand(16, 16, 128);
X = A(:,1);
Y = A(:,2);
Z = A(:,3);



scatter3(X, Y, Z);

B = fftn(A);
scatter3(B(:,1), B(:,2), B(:,3));