@ECHO OFF
set TRAINING_DATA=F:\data learn/stumbleupon/input/train_folds.csv
set MODEL=LogReg
set MODEL_OUTPUT=F:\data learn/stumbleupon/model
set TEST_DATA=F:\data learn/stumbleupon/input/modified_test.csv
SET /A "FOLD=0"
SET /A "LIMIT=4"
:while
if %FOLD% leq %LIMIT% (
   
   python -m src.train
   set /A "FOLD=FOLD+1"
   goto :while
) 

