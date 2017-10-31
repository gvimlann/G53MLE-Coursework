n = 100;
acc1 = zeros([1 n]);
f11 = zeros([1 n]);
esm1 = zeros([1 n]);

for jkjk = 1:n
    multiclass_emotions
    acc1(jkjk) = acc;
    f11(jkjk) = f1;
    esm1(jkjk) = mean_mse;
    clear net;
    jkjk
end

plot([1:n], acc1, [1:n], f11, [1:n], esm1)