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

bool readPGM(const string &filename, 
             simg &image)
{
  ifstream File(filename.c_str(), ifstream::binary);
  int length;
  // Get size
  File.seekg(0,File.end); length=File.tellg(); File.seekg(0,File.beg);
  string dummy;
  File >> dummy >> image.xSize >> image.ySize >> image.maxVal;
  File.get(); // Remove all excessive spaces
  image.init(); // (Re)Initialize image from sizes
  File.read(reinterpret_cast<char*>(image.data.data()), length); 
  return true;
}

bool writePGM(const string &filename, 
              const simg &image)
{
  std::ofstream File(filename.c_str());
  File << "P5\n" << image.xSize << " " << image.ySize << "\n"<< image.maxVal <<"\n";
  File.write (reinterpret_cast<const char*>(image.data.data()), image.size*sizeof(char));
  File.close();
  return true;
}

float dist(simg &image, 
           int patch_radius, 
           int2 p1, // Reference patch
           int2 p2)
{
  float dist(0);
  for(int y=-patch_radius; y<=patch_radius; ++y)
    for(int x=-patch_radius; x<=patch_radius; ++x){
      // Note that image automatically clamps borders
      float tmp = image(p1.x+x, p1.y+y) - image(p2.x+x, p2.y+y);
      dist += tmp*tmp;
    }
  // Normalize
  return dist/patch_radius/patch_radius;
}

void blockMatching(simg &image, 
                   upatches &vec_patches, 
                   upatchnum &num_patches, 
                   int patch_radius,
                   int window_radius, 
                   float sim_th, 
                   int maxN)
{
  int xSize = image.xSize;
  int ySize = image.ySize;
  int step=patch_radius/4; 
  
  // Go through the image with step
  for (int j=0; j<image.ySize; j+=step)
    for (int i=0; i<image.xSize; i+=step)
    {
      // Include (reference patch) self
      vec_patches.push_back( make_int2(i, j) );
      num_patches.push_back(1);
      unsigned curr_i = num_patches.size()-1;

      // Cut boundary of the window if it exceeds image size
      int wxb = max(0, i - window_radius); // window x begin
      int wyb = max(0, j - window_radius); // window y begin
      int wxe = min(xSize-1, i + window_radius); // window x end
      int wye = min(ySize-1, j + window_radius); // window y end 
      
      // Go through the window
      for (int wy = wyb; wy <= wye; wy++)
        for (int wx = wxb; wx <= wxe; wx++)
        {
            // Cap max size in group
            if(num_patches[curr_i]==maxN) break;
            // Exclude itself
            if(i==wx && j==wy) continue;

            float distance = dist(image, patch_radius, make_int2(i,j), make_int2(wx,wy));
            if(distance<sim_th)
            {
              vec_patches.push_back( make_int2(wx, wy) );
              num_patches[curr_i]++;
            }
         }

      // To make cumulative sum, carry value from the previous step
      if(num_patches.size()>1) { // only if prev step happened
        num_patches[curr_i]+=num_patches[curr_i-1];
      }

    } // for j 
  
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

inline pair<int2, int2> getPatchBeginEnd(int2 p, int k, int xSize, int ySize){
  int2 a,b;
  a.x = CLIP(0, p.x - k, xSize);
  a.y = CLIP(0, p.y - k, ySize);
  b.x = CLIP(0, p.x + k, xSize);
  b.y = CLIP(0, p.y + k, ySize);
  return make_pair(a, b);
}

inline void drawGroup(simg &image, upatches &patches, upatchnum &npatches, int k, int xSize, int ySize, int start, int Np){
    uimg img_copy(image.data);
    CImg<unsigned char> img(img_copy.data(),image.xSize,image.ySize,1,1,1);
    const unsigned char c_mat[] = {255, 0, 0};
    for(int i=start; i<Np;++i){
        pair<int2,int2> ref = getPatchBeginEnd(patches[i], k, xSize, ySize); 
        img.draw_rectangle(ref.first.x,ref.first.y,ref.second.x,ref.second.y,c_mat, 0.5);
    }
    img.display(disp1);
}

simg gatherPatches(int idx, 
                   upatches &vec_patches, 
                   upatchnum &num_patches, 
                   simg &image, 
                   int patch_radius){
  int N(0);
  if(idx==0) N=num_patches[idx];
  else N=num_patches[idx]-num_patches[idx-1];

  int patch_size=patch_radius*2+1;
  simg gathered_patches(patch_size, patch_size, N, 0);

  int start = num_patches[idx]-N;
  int end   = num_patches[idx];
  cout << "Start patch "<<start<<" end patch "<<end<<endl;

  for(int z=0;z<gathered_patches.zSize;++z)
  {
    int2 cp = vec_patches[start+z];
    for(int y=0;y<gathered_patches.ySize;++y)
      for(int x=0;x<gathered_patches.zSize;++x){
        gathered_patches(x,y,z) = image(cp.x-patch_radius+x, cp.y-patch_radius+y);
      }
  }
  return gathered_patches;
}
int main(){

  simg image; // Original noisy image 
  cout<<"Reading image..."<<flush;
  if(! readPGM("barbara.pgm", image) ){ cerr << "Failed to open the image.\n"; return EXIT_FAILURE;}
  cout<<"done"<<endl;


  simg denoised(image.xSize, image.ySize, 0); // Denoised image
  simg weights(image.xSize, image.ySize, 0);  // Matrix with accumulated group weights

  int patch_radius(8);    // Patch radius (size=patch_radius*2+1)
  int window_radius(20);  // Search window (size=window_radius*2+1)
  float sim_th(100.0);    // Similarity threshold for the first step
  int maxN(15);           // Maximal number of the patches in one group
  upatches  vec_patches;  // Vector with patch center pixels ((x1,y1), (x2,y2),...)
  // Vector with cumulative sum of patches in each group e.g. (5,8,13)
  // which means fist group has 5 patches, second 3, third 5, etc...
  upatchnum num_patches;    

  cout<<"Performing block matching..."<<flush;
  blockMatching(image, vec_patches, num_patches, patch_radius, window_radius, sim_th, maxN);
  cout<<"done"<<endl;

  cout<<"Performing BM3D..."<<endl;
  cout<<vec_patches.size()<<endl;
  for(unsigned i=0;i<num_patches.size();i++){
    simg group = gatherPatches(i, vec_patches, num_patches, image, patch_radius);
    CImg<unsigned char> test(group.data.data(), group.xSize, group.ySize, group.zSize, 1,1);
    test.display();
    // coeff = waveletGroupTransform(group);
    // th_coeff = thresholdCoeff(coeff, hard_th, group_weight);
    // rec_group = inverseWaveletGroupTransform(th_coeff);
    // aggregate(denoised, weights, vec_patches, num_patches, rec_group, group_weight);
  }
  cout<<"done"<<endl;
  // denoised /= weights;
  //writePGM("denoised.pgm", denoised);

  return EXIT_SUCCESS;
}
