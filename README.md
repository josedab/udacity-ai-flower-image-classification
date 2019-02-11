# Flower classifier project

## Overview
The Flower classifier project is part of the `AI Programming with Python` Nanodegree program.
It consists on applying Transfer learning over a known architecture (such as VGG or Densenet) in order to classify flower images.

## Dependencies
To install these dependencies with pip, you can issue `pip install -r requirements.txt`.

## Usage

### Training
Example of use:
```
python train.py --data_dir './flowers' --arch='vgg16' --save_dir './' --learning_rate 0.001 --hidden_units 5
```
```
00 --gpu --epochs 10
Training model using GPU
Transfer learning process starting
- Loading dataset
- Creating transfer learning model
- Training model
Epoch: 1/10 ---  Training Loss: 4.5241   Validation Loss: 4.4263   Validation Accuracy: 0.0920  
Epoch: 1/10 ---  Training Loss: 4.3233   Validation Loss: 3.9992   Validation Accuracy: 0.1994  
Epoch: 1/10 ---  Training Loss: 3.9725   Validation Loss: 3.6344   Validation Accuracy: 0.2477  
Epoch: 1/10 ---  Training Loss: 3.4935   Validation Loss: 3.0354   Validation Accuracy: 0.3591  
Epoch: 1/10 ---  Training Loss: 3.3410   Validation Loss: 2.7064   Validation Accuracy: 0.4233  
Epoch: 1/10 ---  Training Loss: 2.9503   Validation Loss: 2.5111   Validation Accuracy: 0.4116  
Epoch: 1/10 ---  Training Loss: 2.7216   Validation Loss: 2.1833   Validation Accuracy: 0.4674  
Epoch: 1/10 ---  Training Loss: 2.6785   Validation Loss: 2.0220   Validation Accuracy: 0.4925  
Epoch: 1/10 ---  Training Loss: 2.4526   Validation Loss: 1.8436   Validation Accuracy: 0.5375  
Epoch: 1/10 ---  Training Loss: 2.3576   Validation Loss: 1.7647   Validation Accuracy: 0.5211  
Epoch: 1/10 ---  Training Loss: 2.2228   Validation Loss: 1.4945   Validation Accuracy: 0.5947  
Epoch: 1/10 ---  Training Loss: 2.1641   Validation Loss: 1.4633   Validation Accuracy: 0.5872  
Epoch: 1/10 ---  Training Loss: 1.9855   Validation Loss: 1.3383   Validation Accuracy: 0.6408  
Epoch: 1/10 ---  Training Loss: 1.7683   Validation Loss: 1.3295   Validation Accuracy: 0.6500  
Epoch: 1/10 ---  Training Loss: 1.7327   Validation Loss: 1.2229   Validation Accuracy: 0.6605  
Epoch: 1/10 ---  Training Loss: 1.6465   Validation Loss: 1.2392   Validation Accuracy: 0.6616  
Epoch: 1/10 ---  Training Loss: 1.6383   Validation Loss: 1.2337   Validation Accuracy: 0.6804  
Epoch: 1/10 ---  Training Loss: 1.6286   Validation Loss: 1.1750   Validation Accuracy: 0.6512  
Epoch: 1/10 ---  Training Loss: 1.5494   Validation Loss: 1.1669   Validation Accuracy: 0.6603  
Epoch: 1/10 ---  Training Loss: 1.5168   Validation Loss: 1.0019   Validation Accuracy: 0.7090  
Epoch: 2/10 ---  Training Loss: 0.8161   Validation Loss: 1.0385   Validation Accuracy: 0.7206  
Epoch: 2/10 ---  Training Loss: 1.5290   Validation Loss: 0.9711   Validation Accuracy: 0.7405  
Epoch: 2/10 ---  Training Loss: 1.3511   Validation Loss: 0.9697   Validation Accuracy: 0.7288  
Epoch: 2/10 ---  Training Loss: 1.3157   Validation Loss: 0.8521   Validation Accuracy: 0.7553  
Epoch: 2/10 ---  Training Loss: 1.2058   Validation Loss: 0.9171   Validation Accuracy: 0.7366  
Epoch: 2/10 ---  Training Loss: 1.3185   Validation Loss: 0.9952   Validation Accuracy: 0.7234  
Epoch: 2/10 ---  Training Loss: 1.3003   Validation Loss: 0.9049   Validation Accuracy: 0.7393  
Epoch: 2/10 ---  Training Loss: 1.4582   Validation Loss: 0.9711   Validation Accuracy: 0.7330  
Epoch: 2/10 ---  Training Loss: 1.4177   Validation Loss: 0.9168   Validation Accuracy: 0.7423  
Epoch: 2/10 ---  Training Loss: 1.2598   Validation Loss: 0.8650   Validation Accuracy: 0.7571  
Epoch: 2/10 ---  Training Loss: 1.2975   Validation Loss: 0.8628   Validation Accuracy: 0.7720  
Epoch: 2/10 ---  Training Loss: 1.2762   Validation Loss: 0.8819   Validation Accuracy: 0.7547  
Epoch: 2/10 ---  Training Loss: 1.2740   Validation Loss: 0.8902   Validation Accuracy: 0.7646  
Epoch: 2/10 ---  Training Loss: 1.2818   Validation Loss: 0.7304   Validation Accuracy: 0.7958  
Epoch: 2/10 ---  Training Loss: 1.3364   Validation Loss: 0.7714   Validation Accuracy: 0.7833  
Epoch: 2/10 ---  Training Loss: 1.4652   Validation Loss: 0.7397   Validation Accuracy: 0.7913  
Epoch: 2/10 ---  Training Loss: 1.2195   Validation Loss: 0.7325   Validation Accuracy: 0.7953  
Epoch: 2/10 ---  Training Loss: 1.2516   Validation Loss: 0.7320   Validation Accuracy: 0.7914  
Epoch: 2/10 ---  Training Loss: 1.0585   Validation Loss: 0.6811   Validation Accuracy: 0.8208  
Epoch: 2/10 ---  Training Loss: 1.3765   Validation Loss: 0.6549   Validation Accuracy: 0.8094  
Epoch: 2/10 ---  Training Loss: 1.1998   Validation Loss: 0.7030   Validation Accuracy: 0.8015  
Epoch: 3/10 ---  Training Loss: 1.0401   Validation Loss: 0.7760   Validation Accuracy: 0.7790  
Epoch: 3/10 ---  Training Loss: 1.0453   Validation Loss: 0.6509   Validation Accuracy: 0.8184  
Epoch: 3/10 ---  Training Loss: 1.0721   Validation Loss: 0.7419   Validation Accuracy: 0.8073  
Epoch: 3/10 ---  Training Loss: 1.2632   Validation Loss: 0.7977   Validation Accuracy: 0.7780  
Epoch: 3/10 ---  Training Loss: 1.1062   Validation Loss: 0.7366   Validation Accuracy: 0.8064  
Epoch: 3/10 ---  Training Loss: 1.0814   Validation Loss: 0.6938   Validation Accuracy: 0.8169  
Epoch: 3/10 ---  Training Loss: 1.0440   Validation Loss: 0.6200   Validation Accuracy: 0.8289  
Epoch: 3/10 ---  Training Loss: 0.9234   Validation Loss: 0.6789   Validation Accuracy: 0.8142  
Epoch: 3/10 ---  Training Loss: 1.1727   Validation Loss: 0.6208   Validation Accuracy: 0.8190  
Epoch: 3/10 ---  Training Loss: 1.0728   Validation Loss: 0.5579   Validation Accuracy: 0.8496  
Epoch: 3/10 ---  Training Loss: 1.0369   Validation Loss: 0.6790   Validation Accuracy: 0.8076  
Epoch: 3/10 ---  Training Loss: 1.2384   Validation Loss: 0.6214   Validation Accuracy: 0.8232  
Epoch: 3/10 ---  Training Loss: 1.0957   Validation Loss: 0.6241   Validation Accuracy: 0.8116  
Epoch: 3/10 ---  Training Loss: 1.0939   Validation Loss: 0.6694   Validation Accuracy: 0.8078  
Epoch: 3/10 ---  Training Loss: 1.1558   Validation Loss: 0.6248   Validation Accuracy: 0.8261  
Epoch: 3/10 ---  Training Loss: 1.0271   Validation Loss: 0.5708   Validation Accuracy: 0.8333  
Epoch: 3/10 ---  Training Loss: 0.9878   Validation Loss: 0.7196   Validation Accuracy: 0.8037  
Epoch: 3/10 ---  Training Loss: 1.0556   Validation Loss: 0.6368   Validation Accuracy: 0.8198  
Epoch: 3/10 ---  Training Loss: 1.0548   Validation Loss: 0.6252   Validation Accuracy: 0.8395  
Epoch: 3/10 ---  Training Loss: 1.2056   Validation Loss: 0.5994   Validation Accuracy: 0.8161  
Epoch: 4/10 ---  Training Loss: 0.4902   Validation Loss: 0.6776   Validation Accuracy: 0.8181  
Epoch: 4/10 ---  Training Loss: 1.0293   Validation Loss: 0.6899   Validation Accuracy: 0.7926  
Epoch: 4/10 ---  Training Loss: 0.9641   Validation Loss: 0.6532   Validation Accuracy: 0.8321  
Epoch: 4/10 ---  Training Loss: 0.9909   Validation Loss: 0.5931   Validation Accuracy: 0.8416  
Epoch: 4/10 ---  Training Loss: 0.9592   Validation Loss: 0.6485   Validation Accuracy: 0.8244  
Epoch: 4/10 ---  Training Loss: 0.8746   Validation Loss: 0.6175   Validation Accuracy: 0.8307  
Epoch: 4/10 ---  Training Loss: 0.8555   Validation Loss: 0.5619   Validation Accuracy: 0.8373  
Epoch: 4/10 ---  Training Loss: 0.9293   Validation Loss: 0.5682   Validation Accuracy: 0.8371  
Epoch: 4/10 ---  Training Loss: 0.7355   Validation Loss: 0.6248   Validation Accuracy: 0.8280  
Epoch: 4/10 ---  Training Loss: 1.0139   Validation Loss: 0.6130   Validation Accuracy: 0.8467  
Epoch: 4/10 ---  Training Loss: 1.1700   Validation Loss: 0.6166   Validation Accuracy: 0.8383  
Epoch: 4/10 ---  Training Loss: 1.1018   Validation Loss: 0.5491   Validation Accuracy: 0.8467  
Epoch: 4/10 ---  Training Loss: 1.0548   Validation Loss: 0.5358   Validation Accuracy: 0.8539  
Epoch: 4/10 ---  Training Loss: 0.8418   Validation Loss: 0.6189   Validation Accuracy: 0.8094  
Epoch: 4/10 ---  Training Loss: 0.9617   Validation Loss: 0.6105   Validation Accuracy: 0.8424  
Epoch: 4/10 ---  Training Loss: 1.0914   Validation Loss: 0.5859   Validation Accuracy: 0.8400  
Epoch: 4/10 ---  Training Loss: 0.9141   Validation Loss: 0.5382   Validation Accuracy: 0.8539  
Epoch: 4/10 ---  Training Loss: 1.0310   Validation Loss: 0.5688   Validation Accuracy: 0.8355  
Epoch: 4/10 ---  Training Loss: 0.8192   Validation Loss: 0.5153   Validation Accuracy: 0.8560  
Epoch: 4/10 ---  Training Loss: 0.9503   Validation Loss: 0.5405   Validation Accuracy: 0.8424  
Epoch: 4/10 ---  Training Loss: 0.8330   Validation Loss: 0.5133   Validation Accuracy: 0.8514  
Epoch: 5/10 ---  Training Loss: 0.8424   Validation Loss: 0.5432   Validation Accuracy: 0.8397  
Epoch: 5/10 ---  Training Loss: 0.9096   Validation Loss: 0.4973   Validation Accuracy: 0.8665  
Epoch: 5/10 ---  Training Loss: 0.8689   Validation Loss: 0.5464   Validation Accuracy: 0.8460  
Epoch: 5/10 ---  Training Loss: 0.9716   Validation Loss: 0.5554   Validation Accuracy: 0.8500  
Epoch: 5/10 ---  Training Loss: 0.8961   Validation Loss: 0.5781   Validation Accuracy: 0.8446  
Epoch: 5/10 ---  Training Loss: 0.8859   Validation Loss: 0.5305   Validation Accuracy: 0.8403  
Epoch: 5/10 ---  Training Loss: 0.8808   Validation Loss: 0.5342   Validation Accuracy: 0.8482  
Epoch: 5/10 ---  Training Loss: 0.9323   Validation Loss: 0.5474   Validation Accuracy: 0.8431  
Epoch: 5/10 ---  Training Loss: 0.9082   Validation Loss: 0.5128   Validation Accuracy: 0.8566  
Epoch: 5/10 ---  Training Loss: 0.9470   Validation Loss: 0.6160   Validation Accuracy: 0.8292  
Epoch: 5/10 ---  Training Loss: 0.9218   Validation Loss: 0.5506   Validation Accuracy: 0.8587  
Epoch: 5/10 ---  Training Loss: 0.9220   Validation Loss: 0.4864   Validation Accuracy: 0.8693  
Epoch: 5/10 ---  Training Loss: 0.9793   Validation Loss: 0.5425   Validation Accuracy: 0.8446  
Epoch: 5/10 ---  Training Loss: 0.8019   Validation Loss: 0.5670   Validation Accuracy: 0.8487  
Epoch: 5/10 ---  Training Loss: 0.8844   Validation Loss: 0.5253   Validation Accuracy: 0.8587  
Epoch: 5/10 ---  Training Loss: 0.8455   Validation Loss: 0.5534   Validation Accuracy: 0.8484  
Epoch: 5/10 ---  Training Loss: 0.7557   Validation Loss: 0.5506   Validation Accuracy: 0.8547  
Epoch: 5/10 ---  Training Loss: 0.9100   Validation Loss: 0.5440   Validation Accuracy: 0.8409  
Epoch: 5/10 ---  Training Loss: 0.6570   Validation Loss: 0.5884   Validation Accuracy: 0.8436  
Epoch: 5/10 ---  Training Loss: 1.0641   Validation Loss: 0.5728   Validation Accuracy: 0.8523  
Epoch: 6/10 ---  Training Loss: 0.5182   Validation Loss: 0.5264   Validation Accuracy: 0.8578  
Epoch: 6/10 ---  Training Loss: 0.8283   Validation Loss: 0.5531   Validation Accuracy: 0.8381  
Epoch: 6/10 ---  Training Loss: 0.8770   Validation Loss: 0.5282   Validation Accuracy: 0.8695  
Epoch: 6/10 ---  Training Loss: 0.9311   Validation Loss: 0.5624   Validation Accuracy: 0.8431  
Epoch: 6/10 ---  Training Loss: 0.8559   Validation Loss: 0.5656   Validation Accuracy: 0.8349  
Epoch: 6/10 ---  Training Loss: 0.8030   Validation Loss: 0.5138   Validation Accuracy: 0.8524  
Epoch: 6/10 ---  Training Loss: 0.7748   Validation Loss: 0.4734   Validation Accuracy: 0.8698  
Epoch: 6/10 ---  Training Loss: 0.7913   Validation Loss: 0.5209   Validation Accuracy: 0.8623  
Epoch: 6/10 ---  Training Loss: 0.8849   Validation Loss: 0.5456   Validation Accuracy: 0.8599  
Epoch: 6/10 ---  Training Loss: 0.7675   Validation Loss: 0.5601   Validation Accuracy: 0.8506  
Epoch: 6/10 ---  Training Loss: 0.7468   Validation Loss: 0.5556   Validation Accuracy: 0.8532  
Epoch: 6/10 ---  Training Loss: 0.7957   Validation Loss: 0.5196   Validation Accuracy: 0.8611  
Epoch: 6/10 ---  Training Loss: 1.0252   Validation Loss: 0.5189   Validation Accuracy: 0.8599  
Epoch: 6/10 ---  Training Loss: 0.6669   Validation Loss: 0.4954   Validation Accuracy: 0.8705  
Epoch: 6/10 ---  Training Loss: 0.9000   Validation Loss: 0.5114   Validation Accuracy: 0.8607  
Epoch: 6/10 ---  Training Loss: 0.8523   Validation Loss: 0.5739   Validation Accuracy: 0.8421  
Epoch: 6/10 ---  Training Loss: 0.9316   Validation Loss: 0.5077   Validation Accuracy: 0.8614  
Epoch: 6/10 ---  Training Loss: 1.0028   Validation Loss: 0.5202   Validation Accuracy: 0.8599  
Epoch: 6/10 ---  Training Loss: 0.7707   Validation Loss: 0.5458   Validation Accuracy: 0.8515  
Epoch: 6/10 ---  Training Loss: 0.8097   Validation Loss: 0.5067   Validation Accuracy: 0.8650  
Epoch: 6/10 ---  Training Loss: 0.8044   Validation Loss: 0.5274   Validation Accuracy: 0.8535  
Epoch: 7/10 ---  Training Loss: 0.8557   Validation Loss: 0.5971   Validation Accuracy: 0.8340  
Epoch: 7/10 ---  Training Loss: 0.7239   Validation Loss: 0.5632   Validation Accuracy: 0.8446  
Epoch: 7/10 ---  Training Loss: 0.8529   Validation Loss: 0.5249   Validation Accuracy: 0.8734  
Epoch: 7/10 ---  Training Loss: 0.7584   Validation Loss: 0.5262   Validation Accuracy: 0.8608  
Epoch: 7/10 ---  Training Loss: 0.7297   Validation Loss: 0.5012   Validation Accuracy: 0.8674  
Epoch: 7/10 ---  Training Loss: 0.7272   Validation Loss: 0.5116   Validation Accuracy: 0.8690  
Epoch: 7/10 ---  Training Loss: 0.8906   Validation Loss: 0.4883   Validation Accuracy: 0.8578  
Epoch: 7/10 ---  Training Loss: 0.7464   Validation Loss: 0.4799   Validation Accuracy: 0.8666  
Epoch: 7/10 ---  Training Loss: 0.7479   Validation Loss: 0.4872   Validation Accuracy: 0.8690  
Epoch: 7/10 ---  Training Loss: 0.8648   Validation Loss: 0.5515   Validation Accuracy: 0.8515  
Epoch: 7/10 ---  Training Loss: 0.8160   Validation Loss: 0.5568   Validation Accuracy: 0.8503  
Epoch: 7/10 ---  Training Loss: 0.7839   Validation Loss: 0.5054   Validation Accuracy: 0.8666  
Epoch: 7/10 ---  Training Loss: 0.7023   Validation Loss: 0.4961   Validation Accuracy: 0.8743  
Epoch: 7/10 ---  Training Loss: 0.8118   Validation Loss: 0.4819   Validation Accuracy: 0.8786  
Epoch: 7/10 ---  Training Loss: 0.7241   Validation Loss: 0.5642   Validation Accuracy: 0.8416  
Epoch: 7/10 ---  Training Loss: 0.7860   Validation Loss: 0.5041   Validation Accuracy: 0.8689  
Epoch: 7/10 ---  Training Loss: 0.9155   Validation Loss: 0.5481   Validation Accuracy: 0.8669  
Epoch: 7/10 ---  Training Loss: 0.8139   Validation Loss: 0.5315   Validation Accuracy: 0.8590  
Epoch: 7/10 ---  Training Loss: 0.8745   Validation Loss: 0.4958   Validation Accuracy: 0.8719  
Epoch: 7/10 ---  Training Loss: 0.9250   Validation Loss: 0.4885   Validation Accuracy: 0.8753  
Epoch: 8/10 ---  Training Loss: 0.3667   Validation Loss: 0.4923   Validation Accuracy: 0.8734  
Epoch: 8/10 ---  Training Loss: 0.7574   Validation Loss: 0.4654   Validation Accuracy: 0.8777  
Epoch: 8/10 ---  Training Loss: 0.7410   Validation Loss: 0.5014   Validation Accuracy: 0.8619  
Epoch: 8/10 ---  Training Loss: 0.9187   Validation Loss: 0.5242   Validation Accuracy: 0.8659  
Epoch: 8/10 ---  Training Loss: 0.8629   Validation Loss: 0.4856   Validation Accuracy: 0.8695  
Epoch: 8/10 ---  Training Loss: 0.6717   Validation Loss: 0.4625   Validation Accuracy: 0.8758  
Epoch: 8/10 ---  Training Loss: 0.7567   Validation Loss: 0.4898   Validation Accuracy: 0.8722  
Epoch: 8/10 ---  Training Loss: 0.6260   Validation Loss: 0.5089   Validation Accuracy: 0.8703  
Epoch: 8/10 ---  Training Loss: 0.8813   Validation Loss: 0.5204   Validation Accuracy: 0.8698  
Epoch: 8/10 ---  Training Loss: 0.5997   Validation Loss: 0.5648   Validation Accuracy: 0.8550  
Epoch: 8/10 ---  Training Loss: 0.7693   Validation Loss: 0.5256   Validation Accuracy: 0.8689  
Epoch: 8/10 ---  Training Loss: 0.6395   Validation Loss: 0.4813   Validation Accuracy: 0.8758  
Epoch: 8/10 ---  Training Loss: 0.7131   Validation Loss: 0.4861   Validation Accuracy: 0.8866  
Epoch: 8/10 ---  Training Loss: 0.8395   Validation Loss: 0.4636   Validation Accuracy: 0.8876  
Epoch: 8/10 ---  Training Loss: 0.6777   Validation Loss: 0.4341   Validation Accuracy: 0.8846  
Epoch: 8/10 ---  Training Loss: 0.7886   Validation Loss: 0.4834   Validation Accuracy: 0.8749  
Epoch: 8/10 ---  Training Loss: 0.7331   Validation Loss: 0.4404   Validation Accuracy: 0.8821  
Epoch: 8/10 ---  Training Loss: 0.9220   Validation Loss: 0.4273   Validation Accuracy: 0.8839  
Epoch: 8/10 ---  Training Loss: 0.6955   Validation Loss: 0.4922   Validation Accuracy: 0.8686  
Epoch: 8/10 ---  Training Loss: 0.7880   Validation Loss: 0.5042   Validation Accuracy: 0.8693  
Epoch: 8/10 ---  Training Loss: 0.8415   Validation Loss: 0.4983   Validation Accuracy: 0.8746  
Epoch: 9/10 ---  Training Loss: 0.6811   Validation Loss: 0.4510   Validation Accuracy: 0.8731  
Epoch: 9/10 ---  Training Loss: 0.8056   Validation Loss: 0.4749   Validation Accuracy: 0.8741  
Epoch: 9/10 ---  Training Loss: 0.6411   Validation Loss: 0.5176   Validation Accuracy: 0.8689  
Epoch: 9/10 ---  Training Loss: 0.7912   Validation Loss: 0.4725   Validation Accuracy: 0.8789  
Epoch: 9/10 ---  Training Loss: 0.6924   Validation Loss: 0.5230   Validation Accuracy: 0.8666  
Epoch: 9/10 ---  Training Loss: 0.7140   Validation Loss: 0.5146   Validation Accuracy: 0.8689  
Epoch: 9/10 ---  Training Loss: 0.7340   Validation Loss: 0.4991   Validation Accuracy: 0.8662  
Epoch: 9/10 ---  Training Loss: 0.6187   Validation Loss: 0.5200   Validation Accuracy: 0.8678  
Epoch: 9/10 ---  Training Loss: 0.8513   Validation Loss: 0.5240   Validation Accuracy: 0.8574  
Epoch: 9/10 ---  Training Loss: 0.6897   Validation Loss: 0.5444   Validation Accuracy: 0.8604  
Epoch: 9/10 ---  Training Loss: 0.7727   Validation Loss: 0.4819   Validation Accuracy: 0.8810  
Epoch: 9/10 ---  Training Loss: 0.6696   Validation Loss: 0.4501   Validation Accuracy: 0.8798  
Epoch: 9/10 ---  Training Loss: 0.8389   Validation Loss: 0.4305   Validation Accuracy: 0.8897  
Epoch: 9/10 ---  Training Loss: 0.8320   Validation Loss: 0.4557   Validation Accuracy: 0.8731  
Epoch: 9/10 ---  Training Loss: 0.8385   Validation Loss: 0.4881   Validation Accuracy: 0.8698  
Epoch: 9/10 ---  Training Loss: 0.7892   Validation Loss: 0.4375   Validation Accuracy: 0.8936  
Epoch: 9/10 ---  Training Loss: 0.9552   Validation Loss: 0.4515   Validation Accuracy: 0.8851  
Epoch: 9/10 ---  Training Loss: 0.6781   Validation Loss: 0.4859   Validation Accuracy: 0.8802  
Epoch: 9/10 ---  Training Loss: 0.8650   Validation Loss: 0.4482   Validation Accuracy: 0.8900  
Epoch: 9/10 ---  Training Loss: 0.7905   Validation Loss: 0.5032   Validation Accuracy: 0.8717  
Epoch: 10/10 ---  Training Loss: 0.3353   Validation Loss: 0.4511   Validation Accuracy: 0.8854  
Epoch: 10/10 ---  Training Loss: 0.7632   Validation Loss: 0.4671   Validation Accuracy: 0.8827  
Epoch: 10/10 ---  Training Loss: 0.6682   Validation Loss: 0.4933   Validation Accuracy: 0.8753  
Epoch: 10/10 ---  Training Loss: 0.6472   Validation Loss: 0.4498   Validation Accuracy: 0.8921  
Epoch: 10/10 ---  Training Loss: 0.5817   Validation Loss: 0.4337   Validation Accuracy: 0.8825  
Epoch: 10/10 ---  Training Loss: 0.6271   Validation Loss: 0.4777   Validation Accuracy: 0.8683  
Epoch: 10/10 ---  Training Loss: 0.8009   Validation Loss: 0.4882   Validation Accuracy: 0.8730  
Epoch: 10/10 ---  Training Loss: 0.6388   Validation Loss: 0.4790   Validation Accuracy: 0.8717  
Epoch: 10/10 ---  Training Loss: 0.6678   Validation Loss: 0.4899   Validation Accuracy: 0.8713  
Epoch: 10/10 ---  Training Loss: 0.6793   Validation Loss: 0.4432   Validation Accuracy: 0.8888  
Epoch: 10/10 ---  Training Loss: 0.8604   Validation Loss: 0.4551   Validation Accuracy: 0.8722  
Epoch: 10/10 ---  Training Loss: 0.9162   Validation Loss: 0.5010   Validation Accuracy: 0.8698  
Epoch: 10/10 ---  Training Loss: 0.8016   Validation Loss: 0.5503   Validation Accuracy: 0.8650  
Epoch: 10/10 ---  Training Loss: 0.8277   Validation Loss: 0.4950   Validation Accuracy: 0.8725  
Epoch: 10/10 ---  Training Loss: 0.6964   Validation Loss: 0.4745   Validation Accuracy: 0.8739  
Epoch: 10/10 ---  Training Loss: 0.8663   Validation Loss: 0.4734   Validation Accuracy: 0.8777  
Epoch: 10/10 ---  Training Loss: 0.6483   Validation Loss: 0.4806   Validation Accuracy: 0.8773  
Epoch: 10/10 ---  Training Loss: 0.7241   Validation Loss: 0.5104   Validation Accuracy: 0.8777  
Epoch: 10/10 ---  Training Loss: 0.7043   Validation Loss: 0.4704   Validation Accuracy: 0.8753  
Epoch: 10/10 ---  Training Loss: 0.7900   Validation Loss: 0.5235   Validation Accuracy: 0.8649  
Epoch: 10/10 ---  Training Loss: 0.6053   Validation Loss: 0.4816   Validation Accuracy: 0.8729  
- Saving model into disk
- Model saved at ./checkpoint.pth
Training process completed.
```

### Prediction
```
python predict.py --checkpoint "./checkpoint.pth" --input flowers/train/1/image_06735.jpg
```
```
Prediction results
==================================
Class: pink primrose, Probability: 0.9863
Class: pelargonium, Probability: 0.0098
Class: tree mallow, Probability: 0.0021
Class: hibiscus, Probability: 0.0010
Class: californian poppy, Probability: 0.0003
```
