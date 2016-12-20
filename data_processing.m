 
%  86 = 1; 87 = 2; 88 = 3; 89 =4; 90 =5; 91 = 6; 92 = 7; 93 = 8;

 test_1 = [];
 test_1 = [test_1 VarName5  VarName6  VarName7  VarName8  VarName9 ];
%test_1 = [test_1 VarName0 VarName1 VarName2 VarName3 VarName4 VarName5 VarName6 VarName7 VarName8 VarName9 ];

 test_1 = [test_1 VarName10 VarName11 VarName12 VarName13 VarName14 VarName15 VarName16 VarName17 VarName18 VarName19 ];
 test_1 = [test_1 VarName20 VarName21 VarName22 VarName23 VarName24 VarName25 VarName26 VarName27 VarName28 VarName29 ];
 test_1 = [test_1 VarName30 VarName31 VarName32 VarName33 VarName34 VarName35 VarName36 VarName37 VarName38 VarName39 ];
 test_1 = [test_1 VarName40 VarName41 VarName42 VarName43 VarName44 VarName45 VarName46 VarName47 VarName48 VarName49 ];
 test_1 = [test_1 VarName50 VarName51 VarName52 VarName53 VarName54 VarName55 VarName56 VarName57 VarName58 VarName59 ];
 test_1 = [test_1 VarName60 VarName61 VarName62 VarName63 VarName64 VarName65 VarName66 VarName67 VarName68 VarName69 ];
 test_1 = [test_1 VarName70 VarName71 VarName72 VarName73 VarName74 VarName75 VarName76 VarName77 VarName78 VarName79 ];
 test_1 = [test_1 VarName80 VarName81 VarName82 VarName83 VarName84 VarName85 VarName86 VarName87 VarName88 VarName89 ];
 
 Piston_Rod(1,1) = 5;
 Gear_Part(1,1) = 6;
 Gasket(1,1) = 7;
 Pulley(1,1) = 8;
 
 test_1 = [test_1 Part1 Part2 Part3 Part4 Piston_Rod Gear_Part Gasket Pulley];
 
 %  c = a(find(a(:,3)),:)
 
 Part_1_test =  test_1(find(test_1(:,86)),:);Part_1_test (1,:) = [];Part_1_test(:,86) = 1;
 Part_2_test =  test_1(find(test_1(:,87)),:);Part_2_test (1,:) = [];Part_2_test(:,87) = 0;Part_2_test(:,86) = 2;
 Part_3_test =  test_1(find(test_1(:,88)),:);Part_3_test (1,:) = [];Part_3_test(:,88) = 0;Part_3_test(:,86) = 3;
 Part_4_test =  test_1(find(test_1(:,89)),:);Part_4_test (1,:) = [];Part_4_test(:,89) = 0;Part_4_test(:,86) = 4;
 Part_5_test =  test_1(find(test_1(:,90)),:);Part_5_test (1,:) = [];Part_5_test(:,90) = 0;Part_5_test(:,86) = 5;
 Part_6_test =  test_1(find(test_1(:,91)),:);Part_6_test (1,:) = [];Part_6_test(:,91) = 0;Part_6_test(:,86) = 6;
 Part_7_test =  test_1(find(test_1(:,92)),:);Part_7_test (1,:) = [];Part_7_test(:,92) = 0;Part_7_test(:,86) = 7;
 Part_8_test =  test_1(find(test_1(:,93)),:);Part_8_test (1,:) = [];Part_8_test(:,93) = 0;Part_8_test(:,86) = 8;
 
 test_final = [Part_1_test;Part_2_test;;Part_3_test;Part_4_test;Part_5_test;Part_7_test;Part_8_test];
 
 test_final_label = test_final(:,86);
 test_final(:,86:1:93)=[];
 
 
 
 