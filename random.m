a = randperm(9468);
a = a';
for i = 1:9468
    
    b(i,:) = test_final_all(a(i,1),:);
    b_mean = b - mean(mean(b));
    c(i,:) = test_label_all(a(i,1),:);
    b_min = min(min(b));
    b_max = max(max(b));
    b_normal = (b - b_min)./(b_max - b_min);
end

d = zeros(9468,8);
for i = 1:9468
   d (i,c(i,:)) = 1 ; 
    
    
end

csvwrite('test_final.csv',b);
csvwrite('test_final_center.csv',b_mean);
csvwrite('test_final_nor.csv',b_normal);
csvwrite('test_final_label.csv',d);


