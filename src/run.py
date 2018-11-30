import os,sys
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from models import *





def train_all_models(num_epochs=1000):
    #data genetartor object 
    gen=ImageSequence()
  
    #train CNN
    model_cnn=model_CNN()
    model_cnn.compile(loss=loss_new, optimizer="RMSprop",metrics=[SSIM(kernel_size=100)])
    model_cnn.fit_generator(gen,epochs=num_epochs)
    model_cnn.save_weights("../models/model_cnn.h5")
    
    #train path1	
    path1=model_path1()
    path1.compile(loss=loss_new, optimizer="RMSprop",metrics=[SSIM(kernel_size=100)])
    path1.fit_generator(gen,epochs=num_epochs)
    path1.save_weights("../models/path1.h5")

  
    #train path2	
    path2=model_path2()
    path2.compile(loss=loss_new, optimizer="RMSprop",metrics=[SSIM(kernel_size=100)])
    path2.fit_generator(gen,epochs=num_epochs)
    path2.save_weights("../models/path2.h5")


    #tain morphoN
    morphoN=model_morphoN()
    morphoN.compile(loss=loss_new, optimizer="RMSprop",metrics=[SSIM(kernel_size=100)])
    morphoN.fit_generator(gen,epochs=num_epochs)
    morphoN.save_weights("../models/MorphoN.h5")

    #train morphoN small
    morphoN_small=model_morphoN_small()
    morphoN_small.compile(loss=loss_new, optimizer="RMSprop",metrics=[SSIM(kernel_size=100)])
    morphoN_small.fit_generator(gen,epochs=num_epochs)
    morphoN_small.save_weights("../models/MorphoN_small.h5")


    """
    #test on test data 
    testgen=TestImageSequence()
    t1,t2=gen.get_test_data()
    t2_out=morphoN_small.predict(t1)
    """
   
#get all the models with pretrained weights
def get_all_models():
    model_cnn=model_CNN()
    path1=model_path1()
    path2=model_path2()
    morphoN=model_morphoN()
    morphoN_small=model_morphoN_small()

    model_cnn.load_weights("../models/weights_cnn.h5")
    path1.load_weights("../models/weights_path1.h5")
    path2.load_weights("../models/weights_path2.h5")
    morphoN.load_weights("../models/weights_morphoN.h5")
    morphoN_small.load_weights("../models/weights_morphoN_small.h5")
    D=[model_cnn,path1,path2,morphoN,morphoN_small]
    return D




def save_all_results(output_path="../data/train_data/results/"):
    D=get_all_models()
    gen=ImageSequence()
    t1,t2=gen.get_test_data()

    t2_cnn,t2_path1,t2_path2,t2_path12,t2_small=D[0].predict(t1),D[1].predict(t1),D[2].predict(t1),D[3].predict(t1),D[4].predict(t1)

    t2_cnn[t2_cnn>1]=1
    t2_cnn[t2_cnn<0]=0
    t2_path1[t2_path1>1]=1
    t2_path1[t2_path1<0]=0
    t2_path2[t2_path2>1]=1
    t2_path2[t2_path2<0]=0
    t2_path12[t2_path12>1]=1
    t2_path12[t2_path12<0]=0
    t2_small[t2_small>1]=1
    t2_small[t2_small<0]=0
    L_SCORE=[]
    for j in range(t2.shape[0]):
        sys.stdout.write("\r %d/%d "%(j,t2.shape[0])); sys.stdout.flush()
        """
	#save all the results
        misc.imsave(output_path+str(j)+"_in.png",t1[j,:,:,0])
        misc.imsave(output_path+str(j)+"_gt.png",t2[j,:,:,0])
        misc.imsave(output_path+str(j)+"_"+str(1)+"cnn_out.png",t2_cnn[j,:,:,0])
        misc.imsave(output_path+str(j)+"_"+str(2)+"morphoN.png",t2_path12[j,:,:,0])
        misc.imsave(output_path+str(j)+"_"+str(3)+"path1.png",t2_path1[j,:,:,0])
        misc.imsave(output_path+str(j)+"_"+str(4)+"path2.png",t2_path2[j,:,:,0])
        misc.imsave(output_path+str(j)+"_"+str(4)+"morphoN_small.png",t2_small[j,:,:,0])
        """  
        #calculate SSIM and PSNR
        s1 = ssim(t2_cnn[j,:,:,0],t2[j,:,:,0])
	p1 = psnr(t2_cnn[j,:,:,0],t2[j,:,:,0])
        s2 = ssim(t2_path1[j,:,:,0],t2[j,:,:,0])
	p2 = psnr(t2_path1[j,:,:,0],t2[j,:,:,0])
        s3 = ssim(t2_path2[j,:,:,0],t2[j,:,:,0])
	p3 = psnr(t2_path2[j,:,:,0],t2[j,:,:,0])
        s4 = ssim(t2_path12[j,:,:,0],t2[j,:,:,0])
	p4 = psnr(t2_path12[j,:,:,0],t2[j,:,:,0])
        s5 = ssim(t2_small[j,:,:,0],t2[j,:,:,0])
	p5 = psnr(t2_small[j,:,:,0],t2[j,:,:,0])
        L_SCORE.append([s1,p1,s2,p2,s3,p3,s4,p4,s5,p5])   

    L_SCORE=np.array(L_SCORE)
    print "average SSIM PSNR :",np.mean(L_SCORE,axis=0)



#give input_directory of only rainy images and output_dir 
def main(input_dir="../data/input_images/",output_dir="../data/output/"):
    if(len(os.listdir(input_dir))==0):
	print("ERROR: There is no images in the directory")

    SHAPE_Y,SHAPE_X=512,512
    D_name={-1:"input_",0:"cnn",1:"path1_",2:"path2_",3:"morpho_net_",4:"small_morpho_net_"}

    D=get_all_models()
    for file_name in os.listdir(input_dir):
	img_in=read_image(input_dir+file_name)
	img_shape=img_in.shape
	#if(min(img_t.shape[0],img_t[1])<512):
	if(True):
		plt.imsave(output_dir+file_name.split(".")[0]+D_name[-1]+"."+file_name.split(".")[1],img_in[:,:],cmap="gray")			
		img=resize(img_in,(SHAPE_Y,SHAPE_X,1))
		t1=img[np.newaxis]
		out=[]
		for i in [0,1,2,3,4]:
		    t2=D[i].predict(t1)
		    t2[t2<0]=0
		    t2[t2>1]=1
                    t3=resize(t2[0],(img_shape[:2]))
		    #print t3.shape
		    #plt.imsave(output_dir+D_name[i]+file_name,t3[:,:,0],cmap="gray")			
		    plt.imsave(output_dir+file_name.split(".")[0]+D_name[i]+"."+file_name.split(".")[1],t3[:,:,0],cmap="gray")			
	else:
	    #write code for HR images
	    pass
		



#train all the models
#train_all_models(num_epochs=10000)


#to save te results on test data of rainly dataset, it also calculates the average SSIM/PSNR on every models with pre trained weights
#save_all_results()

#to run it on other dataset
#main(input_dir="../data/input_images/",output_dir="../data/output/")
if(len(sys.argv)==3):
    main(input_dir=sys.argv[1],output_dir=sys.argv[2])
else:
    main(input_dir="../data/input_images/",output_dir="../data/output/")





