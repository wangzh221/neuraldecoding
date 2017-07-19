clear;
rand('seed', 0);
load midterm_train.mat;
[M, C] = size(rate);

mean_pos = mean(kin(:, 1:2));
kin(:, 1:2) = kin(:, 1:2) - ones(M, 1) * mean_pos;
mean_rate = mean(rate);
rate = rate - ones(M, 1) * mean_rate;
 
a = kin(2:M, :)';
b = kin(1:M-1, :)';
A = a * b' * inv(b*b');
W = (a - A*b) * (a - A*b)' / (M - 1);
c = rate';
d = kin';
H = c * d' * inv(d*d');
Q = (c - H*d) * (c - H*d)' / M;
 
load midterm_test;
[M, C] = size(rate);
rate = rate - ones(M, 1) * mean_rate;
d = size(kin, 2);
 
x_m = zeros(d, M);
x = zeros(d, M);
P_m = zeros(d, d, M);
P = zeros(d, d, M);
K = zeros(d, C, M);
tic;
for k = 2:M
    P_m(:, :, k) = A * P(:, :, k-1) * A' + W;
    x_m(:, k) = A * x(:, k-1);
    K(:, :, k) = P_m(:, :, k) * H' * inv(H * P_m(:, :, k) * H' + Q);
    P(:, :, k) = (eye(d) - K(:, :, k) * H) * P_m(:, :, k);
    x(:, k) = x_m(:, k) + K(:, :, k) *(rate(k, :)' - H * x_m(:, k));
end
toc;
x(1:2, :) = x(1:2, :) + mean_pos' * ones(1, M);
 
figure(1);
subplot(2, 1, 1);
plot(1:M, kin(:, 1), 'r--', 1:M, x(1, :)', 'linewidth', 2);
ylabel('x position', 'Fontsize', 10);
xlabel('time(s)');
subplot(2, 1, 2);
plot(1:M, kin(:, 2), 'r--', 1:M, x(2, :)', 'linewidth', 2);
ylabel('y position', 'Fontsize', 10);
xlabel('time(s)');
r2 = 1 - sum((x(1:2, :)' - kin(:, 1:2)).^2) / ...
    sum((kin(:, 1:2) - ones(M, 1) * mean_pos).^2);
r2_com = 1 - sum((x(1:2, :)' - kin(:, 1:2)).^2) ./ ...
    sum((kin(:, 1:2) - ones(M, 1) * mean_pos).^2);
disp(['R2 = ' num2str(r2)]);
 

N = [20, 50, 100, 500];
for i = 1:length(N)
    K = N(i);
    S = zeros(d, K, M);
    weight = ones(K, M) / K;
    Weight = cumsum(weight);
    x_h = zeros(d, M);
 
    tic;
    for t = 2:M
        if mod(t, 100) == 0
            disp(sprintf('t == %d', t));
        end
        r = rand(1, K);
        j = sum(Weight(:, t-1) * ones(1, K) < ones(K, 1) * r) + 1;
        S_p = S(:, j, t-1);
        S(:, :, t) = A * S_p + mvnrnd(zeros(1, d), W, K)';
        for k = 1:K
            log_Like(k, 1) = -0.5 * (log(det(Q)) + (rate(t, :)' - H * S(:, k, t))'...
                * inv(Q) * (rate(t, :)' - H * S(:, k, t)));
        end
        weight(:, t) = ones(K, 1) ./ sum(exp(ones(K, 1) * log_Like' - log_Like * ones(1, K)), 2);
        Weight(:, t) = cumsum(weight(:, t));
        
        x_h(:, t) = S(:, :, t) * weight(:, t);
    end
    toc;
    x_h(1:2, :) = x_h(1:2, :) + mean_pos' * ones(1, M);
    figure(2);clf;
    subplot(2,1,1);
    plot(1:M, kin(:, 1), 'r--', 1:M, x_h(1, :)', 'linewidth', 2);
    ylabel('x position', 'Fontsize', 10);
    xlabel('time(s)');
    subplot(2,1,2);
    plot(1:M, kin(:, 2), 'r--', 1:M, x_h(2, :)', 'linewidth', 2);
    ylabel('y position', 'Fontsize', 10);
    xlabel('time(s)');
    pause;
    R_2(i) = 1- sum((x_h(1:2, :)' - kin(:, 1:2)).^2) / ...
        sum((kin(:, 1:2) - ones(M, 1) * mean_pos).^2);
    R_2_com{i} = 1- sum((x_h(1:2, :)' - kin(:, 1:2)).^2) ./ ...
        sum((kin(:, 1:2) - ones(M, 1) * mean_pos).^2);
end
figure(3);
plot(N, R_2, '--.r', 'Markersize', 25);
hold on;
z = ones(520) * r2;
plot(z, '-b');
xlabel('N');
ylabel('R2');
axis([0, 520, 0.4, 0.8]);
legend('SMC', 'KF');

clear;
rand('seed', 0);
load midterm_train.mat;
[M, C] = size(rate);
d = size(kin, 2);
kin_mean = mean(kin);
kin = kin - ones(M, 1) * kin_mean;
a = kin(2:M, :)';
b = kin(1:M-1, :)';
A = a * b' * inv(b*b');
W = (a - A*b) * (a - A*b)' / (M-1);

for c = 1:C
    alpha_old = zeros(d+1, 1);
    alpha_new = alpha_old + 1;
    count(c) = 0;
    while norm(alpha_new - alpha_old) > 1e-2
        count(c) = count(c) + 1;
        alpha_old = alpha_new;
        Kin = [ones(M, 1) kin];
        nmt = Kin' * rate(:, c) - Kin' * exp(Kin * alpha_old);
        dnt = - Kin' .* (ones(d+1, 1) * exp(Kin * alpha_old)') * Kin;
        alpha_new = alpha_old - dnt\nmt;
    end
    alpha(:, c) = alpha_new;
end
 

load midterm_test.mat
Mt = size(rate, 1);
x_m(:, 1) = zeros(d, 1);
x(:, 1) = x_m(:, 1);
P_m(:, :, 1) = zeros(d);
P(:, :, 1) = zeros(d);
tic;
for k = 2:Mt
    P_m(:, :, k) = A * P(:, :, k-1) * A' + W;
    x_m(:, k) = A * x(:, k-1);
    P(:, :, k) = inv(inv(P_m(:, :, k)) + ...
        alpha(2:end, :) * diag(exp([1 x_m(:, k)'] * alpha)) * alpha(2:end, :)');
    x(:, k) = x_m(:, k) + P(:, :, k) * (alpha(2:end, :) * ...
        (rate(k, :) - exp([1 x_m(:, k)'] * alpha))');
end
toc;
x = x + kin_mean' * ones(1, Mt);
 
r2 = 1 - sum((x' - kin).^2) ./ ...
    sum((kin - ones(Mt, 1) * kin_mean).^2);
disp(['R2 = ' num2str(r2)]);
r2_all = 1 - sum((x' - kin).^2) / ...
    sum((kin - ones(Mt, 1) * kin_mean).^2);
 
figure(1);
y_label = {'x-position', 'y-position', 'x-velocity', 'y-velocity'}; 
for i = 1:d
    subplot(2, 2, i);
    plot(1:Mt, x(i, :), 1:Mt, kin(:, i)');
    xlabel('time(s)');
    ylabel(y_label{i});
end
 
load midterm_test.mat
Mt = size(rate, 1);
N = [20, 50, 100, 500];
for i = 1:length(N)
    K = N(i);
    S = zeros(d, K, Mt);
    weight = ones(K, Mt) / K;
    Weight = cumsum(weight);
    x_h = zeros(d, Mt);
    tic;
    for t = 2:Mt
        if mod(t, 100) == 0
            disp(sprintf('t = %d', t));
        end
        r = rand(1, K);
        j = sum(Weight(:, t-1) * ones(1, K) < ones(K, 1) * r) + 1;
        S_p = S(:, j, t-1);
        S(:, :, t) = A * S_p + mvnrnd(zeros(1, d), W, K)';
        for k = 1:K
            lambda = exp(alpha' * [1; S(:, k, t)]);
            log_Like(k, 1) = sum(log(poisspdf(rate(t, :)', lambda)));
        end
        weight(:, t) = ones(K, 1) ./ sum(exp(ones(K, 1) * log_Like' -...
            log_Like * ones(1, K)), 2);
        Weight(:, t) = cumsum(weight(:, t));
    
        x_h(:, t) = S(:, :, t) * weight(:, t);
    end
    toc,
    x_h = x_h + kin_mean' * ones(1, Mt);   
    R_2{i} = 1 - sum((x_h' - kin).^2) ./ ...
        sum((kin - ones(Mt, 1) * kin_mean).^2);
    disp(['R2 = ' num2str(R_2{i})]);
    R_2_all(i) = 1 - sum((x_h' - kin).^2) / ...
        sum((kin - ones(Mt, 1) * kin_mean).^2);
    figure(2);clf;
    for i = 1:d
        subplot(2, 2, i)
        plot(1:Mt, x_h(i, :), 1:Mt, kin(:, i)');
        xlabel('time(s)');
        ylabel(y_label{i});
    end
    pause;
end
figure(3);
for j = 1:length(N)
    plot(R_2{j}, '--.', 'Markersize', 25);
    xlabel('Components');
    ylabel('Estimate Accuracy R2');
    hold on;
end
hold on;
plot(r2, '-*g', 'linewidth', 2, 'Markersize', 25)
legend('n=20', 'n=50', 'n=100', 'n=500', 'PPF');
figure(4);
plot(N, R_2_all, '--.r', 'Markersize', 25);
hold on;
z = ones(520) * r2_all;
plot(z, '-b');
xlabel('N');
ylabel('R2');
axis([0, 520, 0.4, 0.8]);
legend('SMC', 'PPF');
