#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <vector_types.h>
#include <vector_functions.h>
#include "CImg.h"
#include <omp.h>
using namespace std;
using namespace cimg_library;

#define idx(x,y,x_size) ((x) + (y)*(x_size))
#define idx3(x,y,z,x_size,y_size) ((x) + ((y)+(y_size)*(z))*(x_size))
#define CLIP(minv,val,maxv) (min((maxv), max((minv),(val))))
#define ISQRT2 0.70710678118f

typedef vector<unsigned char> uimg;
typedef vector<int2> upatches;
typedef vector<int> upatchnum;

CImgDisplay disp1, disp2, disp3, disp4;

// Simple image class
class simg{
public:
  inline simg(): xSize(0), ySize(0), zSize(0), maxVal(0) {  }
  inline simg(int width, int height, int depth, unsigned char initialValue): xSize(width), ySize(height), zSize(depth), maxVal(255) {
  size=xSize*ySize*zSize; 
  data.resize(size);
  std::fill(data.begin(), data.end(), initialValue);
  };
  inline simg(int width, int height, unsigned char initialValue): xSize(width), ySize(height), zSize(1), maxVal(255) {
  size=xSize*ySize*zSize; 
  data.resize(size);
  std::fill(data.begin(), data.end(), initialValue);
  };
  inline simg(const simg &in_img, int width, int height, int channels=0, int maxVal=255): xSize(width), ySize(height), zSize(channels), maxVal(maxVal){
    data = in_img.data;
    size = xSize*ySize*zSize;
  };
  inline unsigned char& operator()(const int ax, const int ay, const int az)  {
    int cx = CLIP(0, ax, xSize);
    int cy = CLIP(0, ay, ySize);
    int cz = CLIP(0, az, zSize);
    return data[idx3(cx,cy,cz,xSize,ySize)];
  }
  inline unsigned char& operator()(const int ax, const int ay)  {
    int cx = CLIP(0, ax, xSize);
    int cy = CLIP(0, ay, ySize);
    return data[idx(cx,cy,xSize)];
  }
  void init(){
    if (!xSize) xSize=1;
    if (!ySize) ySize=1;
    if (!zSize) zSize=1;
    if (!maxVal) maxVal=255;
    size = xSize*ySize*zSize;
    data.resize(size);
  };
  uimg data;
  int xSize;
  int ySize;
  int zSize;
  int size;
  int maxVal;
};

bool readPGM(const string &filename, simg &image){
  ifstream File(filename.c_str(), ifstream::binary);
  File.seekg (0, File.end);
  int length = File.tellg();
  File.seekg (0, File.beg);
  string dummy;
  File >> dummy >> image.xSize >> image.ySize >>  image.maxVal;
  File.get();
  image.init();
  File.read(reinterpret_cast<char*>(image.data.data()), length); 
  return true;
}

bool writePGM(const string &filename, const simg &image){
  std::ofstream File(filename.c_str());
  File << "P5\n" << image.xSize << " " << image.ySize << "\n"<< image.maxVal <<"\n";
  File.write (reinterpret_cast<const char*>(image.data.data()), image.size*sizeof(char));
  File.close();
  return true;
}
float dist( const simg &image, int patch_size, int2 p1, int2 p2, int xSize, int ySize){
  float dist(0);
  for(int jj=-patch_size; jj<=patch_size; ++jj)
    for(int ii=-patch_size; ii<=patch_size; ++ii){
      int i1x = CLIP(0, p1.x + ii, xSize);
      int i1y = CLIP(0, p1.y + jj, ySize);
      int i2x = CLIP(0, p2.x + ii, xSize);
      int i2y = CLIP(0, p2.y + jj, ySize);

      float tmp = image.data[ idx(i1x, i1y, xSize)] - image.data[ idx(i2x, i2y, xSize)];
      dist += tmp*tmp;
    }
  return dist/patch_size/patch_size;
}
pair<int2, int2> getPatchBeginEnd(int2 p, int k, int xSize, int ySize){
  int2 a,b;
  a.x = CLIP(0, p.x - k, xSize);
  a.y = CLIP(0, p.y - k, ySize);
  b.x = CLIP(0, p.x + k, xSize);
  b.y = CLIP(0, p.y + k, ySize);
  return make_pair(a, b);
}
void drawGroup(const simg &image, upatches &patches, upatchnum &npatches, int k, int xSize, int ySize, int start, int Np){
    uimg img_copy(image.data);
    CImg<unsigned char> img( img_copy.data(),image.xSize,image.ySize,1,1,1);
    const unsigned char c_mat[] = {255, 0, 0};
    for(int i=start; i<Np;++i){
        pair<int2,int2> ref = getPatchBeginEnd(patches[i], k, xSize, ySize); 
        img.draw_rectangle(ref.first.x,ref.first.y,ref.second.x,ref.second.y,c_mat, 0.5);
    }
    img.display(disp1);
}
void blockMatching( const simg &image, upatches &patches, upatchnum &npatches, int N, float Th, int maxN, int k ){
  int xSize = image.xSize;
  int ySize = image.ySize;
  int step=k; 
  uint ref_patch_count(0);
  //int start(0);
  // Go through the image with step
  for (int j=0; j<image.ySize; j+=step)
    for (int i=0; i<image.xSize; i+=step){
    
    // The reference path is i,j
    ref_patch_count++;
    patches.push_back( make_int2(i, j) );
    npatches.push_back(1); // Allocate memory
    // To make cumulative sum, get the value from the previous step
    if(npatches.size()>=2) npatches[ref_patch_count-1]+=npatches[ref_patch_count-2];

    // Cut boundary of the window if it exceeds image size
    int wxb = max(0, i - N); // window x begin
    int wyb = max(0, j - N); // window y begin
    int wxe = min(xSize - 1, i + N); // window x end
    int wye = min(ySize - 1, j + N); // window y end 
    
    int count_matched=0;
    // Go through the window
    simg distances(N,N,0);
    for (int wy = wyb; wy <= wye; wy++)
      for (int wx = wxb; wx <= wxe; wx++){
        float distance = dist(image, 8, make_int2(i,j), make_int2(wx,wy), xSize, ySize);
        //cout << "Distance: " << distance << endl;
        if (i!=wx && j!=wy && distance<Th && count_matched<=maxN ){
          count_matched++;
          patches.push_back( make_int2(wx, wy) );
          npatches[ref_patch_count-1]++;
        }
       }
    //cout <<"Patch nr: "<<ref_patch_count<<", number in the group: "<<npatches[ref_patch_count-1]<<endl;
    //cout << "Start "<<start<<endl;
    //drawGroup(image, patches, npatches, k, xSize, ySize, start, start+npatches[ref_patch_count-1]);
    //start += npatches[ref_patch_count-1];
  }
  
}
void wavelet2DTransform( simg &coeff, const simg &image){
   simg C(image, image.xSize, image.ySize);
   int xsize=image.xSize;
   int hxsize=image.xSize/2;
   int hysize=image.ySize/2;
   simg CK(hxsize, hysize, 0);
   simg DH(hxsize, hysize, 0);
   simg DV(hxsize, hysize, 0);
   simg DD(hxsize, hysize, 0);
   for(int y=0; y<hysize; y++)
     for(int x=0; x<hxsize; x++) {
       CK.data[idx(x,y,hxsize)]=0.25*(C.data[idx(2*x,2*y,xsize)]+C.data[idx(2*x+1,2*y,xsize)]+C.data[idx(2*x,2*y+1,xsize)]+C.data[idx(2*x+1,2*y+1,xsize)]);
       DH.data[idx(x,y,hxsize)]=0.25*(C.data[idx(2*x,2*y,xsize)]+C.data[idx(2*x,2*y+1,xsize)]-C.data[idx(2*x+1,2*y,xsize)]-C.data[idx(2*x+1,2*y+1,xsize)]);
       DV.data[idx(x,y,hxsize)]=0.25*(C.data[idx(2*x,2*y,xsize)]+C.data[idx(2*x+1,2*y,xsize)]-C.data[idx(2*x,2*y+1,xsize)]-C.data[idx(2*x+1,2*y+1,xsize)]);
       DD.data[idx(x,y,hxsize)]=0.25*(C.data[idx(2*x,2*y,xsize)]-C.data[idx(2*x+1,2*y,xsize)]-C.data[idx(2*x,2*y+1,xsize)]+C.data[idx(2*x+1,2*y+1,xsize)]);
     }
   for (int y=0; y<CK.ySize; y++)
     for (int x=0; x<CK.xSize; x++){
         coeff(x,         y)          = CK(x,y);
         coeff(x+CK.xSize,y)          = DH(x,y);
         coeff(x,         y+CK.ySize) = DV(x,y);
         coeff(x+CK.xSize,y+CK.ySize) = DD(x,y);
       }
   //CImg<unsigned char> sCK(coeff.data.data(), coeff.xSize, coeff.ySize,1,1,1);
   //sCK.display();
}
bool hardThreshold(unsigned char &in, const unsigned char &TH){ if (in<TH) {in = 0; return true;} else { return false; } }
void waveletI2DTransform( simg &image, simg &coeff, const unsigned char &TH, int &retained){
  int hxsize = image.xSize/2;
  int hysize = image.ySize/2;
  simg CK(hxsize, hysize, 0);
  simg DH(hxsize, hysize, 0);
  simg DV(hxsize, hysize, 0);
  simg DD(hxsize, hysize, 0);
  int thresholded_count(0);
  for (int y = 0; y < CK.ySize; y++)
     for (int x = 0; x < CK.xSize; x++){
       CK(x,y)=coeff(x,y);
       unsigned char DHv = coeff(x+CK.xSize,y);
       unsigned char DVv = coeff(x,y+CK.xSize);
       unsigned char DDv = coeff(x+CK.xSize,y+CK.xSize);
       if(hardThreshold(DHv,TH)) thresholded_count++;
       if(hardThreshold(DVv,TH)) thresholded_count++;
       if(hardThreshold(DDv,TH)) thresholded_count++;
       DH(x,y)=DHv;
       DV(x,y)=DVv;
       DD(x,y)=DDv;
     }
   for (int y = 0; y < CK.ySize; y++)
     for (int x = 0; x < CK.xSize; x++) 
     {
        image(2*x,2*y)    =CK(x,y)+DH(x,y)+DV(x,y)+DD(x,y);
        image(2*x+1,2*y)  =CK(x,y)-DH(x,y)+DV(x,y)-DD(x,y);
        image(2*x,2*y+1)  =CK(x,y)+DH(x,y)-DV(x,y)-DD(x,y);
        image(2*x+1,2*y+1)=CK(x,y)-DH(x,y)-DV(x,y)+DD(x,y);
     }
   // Count of non-zeroed coefficients (excluding CK matrix)
   retained=(image.size-CK.size-thresholded_count);
   cout<<"Retained: "<<retained<<" coeffs, out of "<<image.size-CK.size<<endl;
   CImg<unsigned char> sCK(image.data.data(), image.xSize, image.ySize,1,1,1);
   sCK.display();
}
void wavelet1DTransform(simg &coeff, simg &image, int dim){
  int xsize = image.xSize;
  int ysize = image.ySize;
  int hxsize = image.xSize/2;
  int hysize = image.ySize/2;
  int dimxsize(0),dimysize(0);
  if (dim==1){
     dimxsize=hxsize;
     dimysize=ysize;
  } else if (dim==2){
     dimxsize=xsize;
     dimysize=hysize;
  }
  simg CK (dimxsize, dimysize, 0);
  simg DK (dimxsize, dimysize, 0);
  for(int y=0; y<dimysize; ++y)
    for(int x=0; x<dimxsize; ++x){
      // ...
      if (dim==1){
        DK(x,y) = 0.5*(image(2*x,y) - image(2*x+1,y));
        CK(x,y) = 0.5*(image(2*x,y) + image(2*x+1,y));
      } else if(dim==2){
        DK(x,y) = 0.5*(image(x,2*y) - image(x,2*y+1));
        CK(x,y) = 0.5*(image(x,2*y) + image(x,2*y+1));
      }
    }
   // Write coeff
   for(int y=0; y<dimysize; ++y)
     for(int x=0; x<dimxsize; ++x){
       if (dim==1){
          coeff(x+dimxsize,y)=DK(x,y);
          coeff(x,y)=CK(x,y);
        } else if(dim==2){
          coeff(x,y+dimysize)=DK(x,y);
          coeff(x,y)=CK(x,y);
        }
     }
   //CImg<unsigned char> sDK(DK.data.data(), DK.xSize, DK.ySize,1,1,1);
   //sDK.display();
}
void waveletI1DTransform(simg &image, simg &coeff, int dim){
  int xsize = coeff.xSize;
  int ysize = coeff.ySize;
  int hxsize = coeff.xSize/2;
  int hysize = coeff.ySize/2;
  int dimxsize(0),dimysize(0);
  if (dim==1){
     dimxsize=hxsize;
     dimysize=ysize;
  } else if (dim==2){
     dimxsize=xsize;
     dimysize=hysize;
  }
  for(int y=0; y<dimysize; ++y)
    for(int x=0; x<dimxsize; ++x){
      if (dim==1){
        // C           =     CK           DK
        unsigned char DK = coeff(x+dimxsize, y);
        image(2*x,y)   =coeff(x,y) + DK;
        image(2*x+1,y) =coeff(x,y) - DK;
      } else if(dim==2){
        // C           =     CK           DK
        unsigned char DK = coeff(x, y+dimysize);
        image(x,2*y)   =coeff(x,y) + DK;
        image(x,2*y+1) =coeff(x,y) - DK;
      }
    }
}
simg gatherPatches(int idx, upatches &patches, upatchnum &nump, simg &image, int patchSize){
  int num_patches(0);
  if(idx==0) num_patches=nump[idx];
  else num_patches=nump[idx]-nump[idx-1];

  simg gathered_patches(patchSize*2+1,patchSize*2+1,num_patches, 0);
  cout << gathered_patches.xSize<<endl;
  cout << gathered_patches.ySize<<endl;
  cout << gathered_patches.zSize<<endl;
  cout << "Start patch "<<nump[idx]-num_patches<<" end patch "<<nump[idx]<<endl;
  for(int z=nump[idx]-num_patches;z<nump[idx];++z){
   int2 cp = patches[z];
    for(int y=0;y<gathered_patches.ySize;++y)
      for(int x=0;x<gathered_patches.zSize;++x){
        gathered_patches(x,y,idx) = image(cp.x-patchSize, cp.y-patchSize);
      }}
  return gathered_patches;
}
int main(){
  simg in_image;      // Original noisy image 
  int k(4);           // Patch size
  int N(20);          // Search window
  float Th(500.0);    // Similarity threshold for the first step
  int maxN(15);       // Maximal number of the patches in one group
  upatches patches;   // Vector 
  upatchnum npatches; // Vector 
  cout<<"Reading image..."<<flush;
  if(! readPGM("barbara.pgm", in_image) ){ cerr << "Failed to open the image.\n"; return EXIT_FAILURE;}
  cout<<"done"<<endl;

  CImg<unsigned char> orig(in_image.data.data(), in_image.xSize, in_image.ySize,1,1,1); 
  
  cout<<"Performing block matching..."<<flush;
  blockMatching(in_image, patches, npatches, N, Th, maxN, k);
  cout<<"done"<<endl;
  cout<<"Gathering patches together..."<<flush;
  //for(unsigned i=0;i<npatches.size();i++)
    simg group = gatherPatches(1, patches, npatches, in_image, k);
  cout<<"done"<<endl;
  exit(0);
  cout<<"Performing wavelet2DTransform..."<<flush;
  simg coeff(in_image.xSize, in_image.ySize, 0);
  wavelet2DTransform(coeff, in_image);
  cout<<"done"<<endl;
  int retained(0);
  waveletI2DTransform(in_image, coeff, 0, retained);
  CImg<unsigned char> abc(in_image.data.data(), in_image.xSize, in_image.ySize,1,1,1); 

  simg coeffx(coeff.xSize,coeff.ySize,0);
  simg coeffy(coeff.xSize,coeff.ySize,0);
  wavelet1DTransform(coeffx, in_image, 1);
  wavelet1DTransform(coeffy, in_image, 2);

  simg coeffxy(coeff.xSize,coeff.ySize,0);
  simg coeffyx(coeff.xSize,coeff.ySize,0);
  wavelet1DTransform(coeffxy, coeffx, 2);
  wavelet1DTransform(coeffyx, coeffy, 1);

  simg one(coeff.xSize,coeff.ySize,0);
  simg two(coeff.xSize,coeff.ySize,0);
  waveletI1DTransform(one,coeffxy, 2);
  waveletI1DTransform(two,one, 1);
  CImg<unsigned char> onedinverse(two.data.data(), coeff.xSize, coeff.ySize,1,1,1); onedinverse.display();
  CImg<unsigned char> diff6=orig-onedinverse; diff6.display();

  CImg<unsigned char> debug0(coeff.data.data(), coeff.xSize, coeff.ySize,1,1,1); debug0.display(disp1);
  CImg<unsigned char> debug3(coeffxy.data.data(), coeff.xSize, coeff.ySize,1,1,1); debug3.display(disp2);
  CImg<unsigned char> debug4(coeffyx.data.data(), coeff.xSize, coeff.ySize,1,1,1); debug4.display(disp3);
  cout << "Difference between two" <<endl;
  CImg<unsigned char> diff = (debug3-debug4); diff.display();
  simg simgxy(coeff.xSize, coeff.ySize, 0), simgyx(coeff.xSize, coeff.ySize,0);
  waveletI2DTransform(simgxy, coeffxy, 0, retained);
  CImg<unsigned char> imgxy(simgxy.data.data(), coeff.xSize, coeff.ySize,1,1,1); 
  waveletI2DTransform(simgyx, coeffyx, 0, retained);
  CImg<unsigned char> imgyx(simgyx.data.data(), coeff.xSize, coeff.ySize,1,1,1);
  cout << "Difference between orig and imgxy" <<endl;
  CImg<unsigned char> diff1 =orig-imgxy; diff1.display();
  cout << "Difference between orig and imgyx" <<endl;
  CImg<unsigned char> diff2=orig-imgyx; diff2.display();
  cout << "Difference between orig and 2D tranform" <<endl;
  CImg<unsigned char> diff3=orig-abc; diff3.display();
  cout << "Difference between 2D and imgxy" <<endl;
  CImg<unsigned char> diff4=debug0-debug3; diff4.display();
  cout << "Difference between 2D and imgyx" <<endl;
  CImg<unsigned char> diff5=debug0-debug4; diff5.display();
  cout << "Difference thresholded 2D and 1D" <<endl;
  CImg<unsigned char> diff7=abc-onedinverse; diff7.display();
  //simg coeff2(coeff, coeff.xSize, coeff.ySize);
  //wavelet1DTransform(coeff, coeff2, 2);
  // tranformThresholdingITransform
  // Aggregation
  writePGM("denoised.pgm", in_image);

  return EXIT_SUCCESS;
}
