#include <cassert>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <vector_types.h>
#include <vector_functions.h>
#include "CImg.h"
#include <omp.h>
#include <random>
#include <limits>
using namespace std;
using namespace cimg_library;

#define idx(x,y,x_size) ((x) + (y)*(x_size))
#define idx3(x,y,z,x_size,y_size) ((x) + ((y)+(y_size)*(z))*(x_size))
#define CLIP(minv,val,maxv) (min((maxv), max((minv),(val))))
#define ISQRT2 0.707106781186547524400844362104849039284835937688474036588f

typedef vector<unsigned char> uimg;
typedef pair<float, int2> udist;
typedef vector<udist> udistvec;
typedef vector<int2> upatches;
typedef vector<int> upatchnum;

CImgDisplay disp1, disp2, disp3, disp4;

struct Parameters{
  string filename = "barbara.pgm";
  int patch_radius=4;   // Patch radius (size=patch_radius*2)
  int window_radius=20; // Search window (size=window_radius*2+1)
  float sim_th=2500.0;   // Similarity threshold for the first step
  int maxN=16;          // Maximal number of the patches in one group
  float hard_th=200.0; // Hard schrinkage threshold
  float sigma=25.0;
  float noise_sigma=25.0;
  float garotte=false;
  bool add_noise_ = false;
};

// Simple image class
template<typename T>
class SImg{
public:
  inline SImg(): xSize(0), ySize(0), zSize(0), maxVal(0) {  }
  inline SImg(int width, int height, int depth, T initialValue): xSize(width), ySize(height), zSize(depth), maxVal(255) {
  size=xSize*ySize*zSize;
  data.resize(size);
  std::fill(data.begin(), data.end(), initialValue);
  };
  inline SImg(int width, int height, T initialValue): xSize(width), ySize(height), zSize(1), maxVal(255) {
  size=xSize*ySize*zSize;
  data.resize(size);
  std::fill(data.begin(), data.end(), initialValue);
  };
  inline SImg(const SImg<T> &in_img, int width, int height, int channels=0, T maxVal=255): xSize(width), ySize(height), zSize(channels), maxVal(maxVal){
    data = in_img.data;
    size = xSize*ySize*zSize;
  };
  inline T& operator()(const int ax, const int ay, const int az)  {
    int cx = CLIP(0, ax, xSize);
    int cy = CLIP(0, ay, ySize);
    int cz = CLIP(0, az, zSize);
    return data[idx3(cx,cy,cz,xSize,ySize)];
  }
  inline T& operator()(const int ax, const int ay)  {
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
  void normalize(T n_min, T n_max){
    T c_min(numeric_limits<T>::max()), c_max(numeric_limits<T>::min());
    for(int i=0;i<data.size();++i){
      if( data[i] < c_min ) c_min = data[i];
      if( data[i] > c_max ) c_max = data[i];
    }

    for(int i=0;i<data.size();++i){
      data[i] = ((data[i]-c_min)/(c_max-c_min))*n_max;
    }
  }
  vector<T> data;
  int xSize;
  int ySize;
  int zSize;
  int size;
  int maxVal;
};

bool readPGM(const string &filename,
             SImg<unsigned char> &image)
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
              const SImg<unsigned char> &image)
{
  std::ofstream File(filename.c_str());
  File << "P5\n" << image.xSize << " " << image.ySize << "\n"<< image.maxVal <<"\n";
  File.write (reinterpret_cast<const char*>(image.data.data()), image.size*sizeof(char));
  File.close();
  return true;
}

bool comp2float(const float &v1, const float &v2){
  return fabs(v1-v2) < 0.0001f;

}

float dist(SImg<float> &image,
           int patch_radius,
           int2 p1, // Reference patch
           int2 p2)
{
  float dist(0);
  for(int y=-patch_radius; y<patch_radius; ++y)
    for(int x=-patch_radius; x<patch_radius; ++x){
      // Note that image automatically clamps borders
      float tmp = image(p1.x+x, p1.y+y) - image(p2.x+x, p2.y+y);
      dist += tmp*tmp;
    }
  // Normalize
  return dist/static_cast<float>(patch_radius*patch_radius);
}

bool sort_distances(udist v1, udist v2) { return v1.first<v2.first; }

inline bool powerOfTwo(const int x){ return !(x == 0) && !(x & (x - 1)); }
inline int closestPowerOfTwo(const int x){
 int power = 1;
 while(power < x) power*=2;
 return power/2;
}

void blockMatching(SImg<float> &image,
                   upatches &vec_patches,
                   upatchnum &num_patches,
                   int patch_radius,
                   int window_radius,
                   float sim_th,
                   int maxN)
{
  int xSize = image.xSize;
  int ySize = image.ySize;
  int step=1;
  unsigned curr_i = 0;

  // Go through the image with step
  for (int j=0; j<image.ySize; j+=step)
    for (int i=0; i<image.xSize; i+=step)
    {
      // Include (reference patch) self
      // init
      num_patches.push_back(0);
      curr_i = num_patches.size()-1;

      // Cut boundary of the window if it exceeds image size
      int wxb = max(0, i - window_radius); // window x begin
      int wyb = max(0, j - window_radius); // window y begin
      int wxe = min(xSize-1, i + window_radius); // window x end
      int wye = min(ySize-1, j + window_radius); // window y end
      int wins=1;

      int win_xsize=(wxe-wxb+1);
      int win_ysize=(wye-wyb+1);

      int win_size=win_xsize*win_ysize;
      udistvec distances(win_size);

      // Go through the window
#pragma omp parallel for
      for (int wy=wyb; wy<=wye; wy+=wins)
        for (int wx=wxb; wx<=wxe; wx+=wins)
        {

            distances[idx(wx-wxb,wy-wyb,win_xsize)].first = dist(image, patch_radius, make_int2(i,j), make_int2(wx,wy));
            distances[idx(wx-wxb,wy-wyb,win_xsize)].second = make_int2(wx,wy);

         }

      sort(distances.begin(), distances.end(), sort_distances);

      for(unsigned i=0; i<distances.size();++i){
        if (distances[i].first>sim_th || num_patches[curr_i]>=maxN){ break; }
        vec_patches.push_back(distances[i].second);
        num_patches[curr_i]++;
      }

      if( !powerOfTwo(num_patches[curr_i])){
        int new_size = closestPowerOfTwo( num_patches[curr_i] );
        int diff = num_patches[curr_i]-new_size;
        vec_patches.erase(vec_patches.end()-diff, vec_patches.end());
        num_patches[curr_i]-=diff;
      }

      // To make cumulative sum, carry value from the previous step
      if(num_patches.size()>1) { // only if prev step happened
        num_patches[curr_i]+=num_patches[curr_i-1];
      }

    } // for j

}

void wavelet2DTransform( SImg<float> &coeff, const SImg<float> &image){
   SImg<float> C(image, image.xSize, image.ySize);
   int xsize=image.xSize;
   int hxsize=image.xSize/2;
   int hysize=image.ySize/2;
   SImg<float> CK(hxsize, hysize, 0);
   SImg<float> DH(hxsize, hysize, 0);
   SImg<float> DV(hxsize, hysize, 0);
   SImg<float> DD(hxsize, hysize, 0);
   for(int y=0; y<hysize; y++)
     for(int x=0; x<hxsize; x++) {
       CK.data[idx(x,y,hxsize)]=0.5*(C.data[idx(2*x,2*y,xsize)]+C.data[idx(2*x+1,2*y,xsize)]+C.data[idx(2*x,2*y+1,xsize)]+C.data[idx(2*x+1,2*y+1,xsize)]);
       DH.data[idx(x,y,hxsize)]=0.5*(C.data[idx(2*x,2*y,xsize)]+C.data[idx(2*x,2*y+1,xsize)]-C.data[idx(2*x+1,2*y,xsize)]-C.data[idx(2*x+1,2*y+1,xsize)]);
       DV.data[idx(x,y,hxsize)]=0.5*(C.data[idx(2*x,2*y,xsize)]+C.data[idx(2*x+1,2*y,xsize)]-C.data[idx(2*x,2*y+1,xsize)]-C.data[idx(2*x+1,2*y+1,xsize)]);
       DD.data[idx(x,y,hxsize)]=0.5*(C.data[idx(2*x,2*y,xsize)]-C.data[idx(2*x+1,2*y,xsize)]-C.data[idx(2*x,2*y+1,xsize)]+C.data[idx(2*x+1,2*y+1,xsize)]);
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
void waveletI2DTransform( SImg<float> &image, SImg<float> &coeff){
  int hxsize = image.xSize/2;
  int hysize = image.ySize/2;
  SImg<float> CK(hxsize, hysize, 0);
  SImg<float> DH(hxsize, hysize, 0);
  SImg<float> DV(hxsize, hysize, 0);
  SImg<float> DD(hxsize, hysize, 0);
  for (int y = 0; y < CK.ySize; y++)
     for (int x = 0; x < CK.xSize; x++){
       CK(x,y)=coeff(x,y);
       float DHv = coeff(x+CK.xSize,y);
       float DVv = coeff(x,y+CK.xSize);
       float DDv = coeff(x+CK.xSize,y+CK.xSize);
       DH(x,y)=DHv;
       DV(x,y)=DVv;
       DD(x,y)=DDv;
     }
   for (int y = 0; y < CK.ySize; y++)
     for (int x = 0; x < CK.xSize; x++)
     {
        image(2*x,2*y)    =ISQRT2*(CK(x,y)+DH(x,y)+DV(x,y)+DD(x,y));
        image(2*x+1,2*y)  =ISQRT2*(CK(x,y)-DH(x,y)+DV(x,y)-DD(x,y));
        image(2*x,2*y+1)  =ISQRT2*(CK(x,y)+DH(x,y)-DV(x,y)-DD(x,y));
        image(2*x+1,2*y+1)=ISQRT2*(CK(x,y)-DH(x,y)-DV(x,y)+DD(x,y));
     }
}
void wavelet1DTransform(SImg<float> &coeff, SImg<float> &image, int dim){
  int dimxsize = image.xSize;
  int dimysize = image.ySize;
  int dimzsize = image.zSize;
  if (dim==1){
     dimxsize/=2;
  } else if (dim==2){
     dimysize/=2;
  } else if (dim==3){
     dimzsize/=2;
  }
  SImg<float> CK (dimxsize, dimysize, dimzsize, 0);
  SImg<float> DK (dimxsize, dimysize, dimzsize, 0);
  for(int z=0; z<dimzsize; ++z)
    for(int y=0; y<dimysize; ++y)
      for(int x=0; x<dimxsize; ++x){
        if(dim==1){
          DK(x,y,z) = ISQRT2*(image(x*2,y,z) - image(x*2+1,y,z));
          CK(x,y,z) = ISQRT2*(image(x*2,y,z) + image(x*2+1,y,z));
        } else if(dim==2){
          DK(x,y,z) = ISQRT2*(image(x,y*2,z) - image(x,y*2+1,z));
          CK(x,y,z) = ISQRT2*(image(x,y*2,z) + image(x,y*2+1,z));
        } else if(dim==3){
          DK(x,y,z) = ISQRT2*(image(x,y,z*2) - image(x,y,z*2+1));
          CK(x,y,z) = ISQRT2*(image(x,y,z*2) + image(x,y,z*2+1));
        }
    }
 // Write coeff
 for(int z=0; z<dimzsize; ++z)
   for(int y=0; y<dimysize; ++y)
     for(int x=0; x<dimxsize; ++x){
       if (dim==1){
          coeff(x+dimxsize,y,z)=DK(x,y,z);
          coeff(x,y,z)=CK(x,y,z);
        } else if(dim==2){
          coeff(x,y+dimysize,z)=DK(x,y,z);
          coeff(x,y,z)=CK(x,y,z);
        } else if(dim==3){
            coeff(x,y,z+dimzsize)=DK(x,y,z);
            coeff(x,y,z)=CK(x,y,z);
          }
     }
}
void waveletI1DTransform(SImg<float> &image, SImg<float> &coeff, int dim){
  int dimxsize = coeff.xSize;
  int dimysize = coeff.ySize;
  int dimzsize = coeff.zSize;
  if (dim==1){
     dimxsize/=2;
  } else if (dim==2){
     dimysize/=2;
  } else if (dim==3){
     dimzsize/=2;
  }
  for(int z=0; z<dimzsize; ++z)
    for(int y=0; y<dimysize; ++y)
      for(int x=0; x<dimxsize; ++x){
          // C           =     CK                            DK
        if (dim==1){
          image(2*x,y,z)   =ISQRT2*(coeff(x,y,z) + coeff(x+dimxsize,y,z));
          image(2*x+1,y,z) =ISQRT2*(coeff(x,y,z) - coeff(x+dimxsize,y,z));
        } else if(dim==2){
          image(x,2*y,z)   =ISQRT2*(coeff(x,y,z) + coeff(x,y+dimysize,z));
          image(x,2*y+1,z) =ISQRT2*(coeff(x,y,z) - coeff(x,y+dimysize,z));
        } else if (dim==3){
          image(x,y,2*z)   =ISQRT2*(coeff(x,y,z) + coeff(x,y,z+dimzsize));
          image(x,y,2*z+1) =ISQRT2*(coeff(x,y,z) - coeff(x,y,z+dimzsize));
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

inline void drawGroup(SImg<float> &image, upatches &patches, upatchnum &npatches, int k, int xSize, int ySize, int start, int Np){
    vector<float> img_copy(image.data);
    CImg<float> img(img_copy.data(),image.xSize,image.ySize,1,1,1);
    const unsigned char c_mat[] = {255, 0, 0};
    for(int i=start; i<Np;++i){
        pair<int2,int2> ref = getPatchBeginEnd(patches[i], k, xSize, ySize);
        img.draw_rectangle(ref.first.x,ref.first.y,ref.second.x,ref.second.y,c_mat, 0.5);
    }
    img.display(disp1);
}

SImg<float> gatherPatches(int idx,
                   upatches &vec_patches,
                   upatchnum &num_patches,
                   SImg<float> &image,
                   int patch_radius){
  int N(0);
  if(idx==0) N=num_patches[idx];
  else N=num_patches[idx]-num_patches[idx-1];
  int append=0;
  if(N%2) append=1;

  int patch_size=patch_radius*2;
  SImg<float> gathered_patches(patch_size, patch_size, N+append, 0);

  int start=num_patches[idx]-N;
  for(int z=0;z<gathered_patches.zSize;++z)
  {
    int2 cp = vec_patches[start+z];
    for(int y=0;y<gathered_patches.ySize;++y)
      for(int x=0;x<gathered_patches.xSize;++x){
        if (z<N)
          gathered_patches(x,y,z) = image(cp.x-patch_radius+x, cp.y-patch_radius+y);
        else
          gathered_patches(x,y,z) = 0;
      }
  }
  return gathered_patches;
}
SImg<float> waveletGroupTransform(SImg<float> &group){
  SImg<float> res(group.xSize, group.ySize, group.zSize, 0);
  SImg<float> coeff1(group.xSize, group.ySize, group.zSize, 0);
  SImg<float> coeff2(group.xSize, group.ySize, group.zSize, 0);

  wavelet1DTransform(coeff1, group, 1);
  wavelet1DTransform(coeff2, coeff1, 2);
  if (coeff2.zSize>1){
    wavelet1DTransform(res, coeff2, 3);
  }
  else
   res.data = coeff2.data;

  return res;
}
SImg<float> inverseWaveletGroupTransform(SImg<float> &coeff){
  SImg<float> res(coeff.xSize, coeff.ySize, coeff.zSize, 0);
  SImg<float> coeff1(coeff.xSize, coeff.ySize, coeff.zSize, 0);
  SImg<float> coeff2(coeff.xSize, coeff.ySize, coeff.zSize, 0);

  if (coeff.zSize>1){
    waveletI1DTransform(coeff1, coeff, 3);
  }
  else
    coeff1.data = coeff.data;
  waveletI1DTransform(coeff2, coeff1, 2);
  waveletI1DTransform(res, coeff2, 1);

  return res;
}
template<typename T>
void assertGroups(SImg<T> &group1, SImg<T> &group2){
  assert(group1.xSize==group2.xSize);
  assert(group1.ySize==group2.ySize);
  assert(group1.zSize==group2.zSize);
  for(int z=0;z<group1.zSize;++z)
    for(int y=0;y<group1.ySize;++y)
      for(int x=0;x<group1.xSize;++x){
        if ( !comp2float(group1(x,y,z), group2(x,y,z)) )
          cout<<fixed<<setprecision(6)<<"x "<<x<<" y "<<y<<" z "<<z<<" val1 "<<group1(x,y,z)<<" val2 "<<group2(x,y,z)<<endl;
        assert( comp2float(group1(x,y,z), group2(x,y,z)) );
      }
  cout << "Assertion passed." << endl;
}
void simgUnsignedToFloat(SImg<float> &dst, SImg<unsigned char> &src)
{
  for(int z=0;z<dst.zSize;++z)
    for(int y=0;y<dst.ySize;++y)
      for(int x=0;x<dst.xSize;++x){
        dst(x,y,z)=static_cast<float>(src(x,y,z));
      }
}
void simgFloatToUnsigned(SImg<unsigned char> &dst, SImg<float> &src){
  for(int z=0;z<dst.zSize;++z)
    for(int y=0;y<dst.ySize;++y)
      for(int x=0;x<dst.xSize;++x){
        dst(x,y,z)=static_cast<unsigned char>(round(src(x,y,z)));
      }
}
void thresholdCoeff( SImg<float> &coeff, Parameters &p, float &group_weight){
  int count(0);
  for(int z=0;z<coeff.zSize;++z)
    for(int y=0;y<coeff.ySize;++y)
      for(int x=0;x<coeff.xSize;++x){
        if (z<coeff.zSize/2 && y<coeff.ySize/2 && x<coeff.xSize/2) continue;

        float val = coeff(x,y,z);
        if( fabs(val)<p.hard_th ){
          // Hard thresholding
          coeff(x,y,z) = 0;
        } else {
          if(p.garotte)
            coeff(x,y,z) = val-((p.hard_th*p.hard_th)/val);
          count++;
        }
      }
  if (count){
    group_weight=1./(count*p.sigma*p.sigma);
  } else {
    group_weight=1;
  }
}
void aggregate(int idx, SImg<float> &denoised, SImg<float> &weights,const upatches &vec_patches,const upatchnum &num_patches, SImg<float> &group, const float &group_weight, int patch_radius){
  int N(0);
  if(idx==0) N=num_patches[idx];
  else N=num_patches[idx]-num_patches[idx-1];
  int start = num_patches[idx]-N;

  for(int z=start;z<num_patches[idx];++z)
  {
      int2 cp = vec_patches[z];
      for(int y=-patch_radius;y<patch_radius;++y)
        for(int x=-patch_radius;x<patch_radius;++x){
            denoised(cp.x+x,cp.y+y) += group_weight*group(x+patch_radius, y+patch_radius, z-start);
            weights(cp.x+x,cp.y+y) += group_weight;
        }
  }
}
void add_noise(SImg<float> &dst, SImg<float> &src, const float sigma){
  default_random_engine generator;
  normal_distribution<double> dist(0, sigma);
  for(unsigned i=0; i<dst.data.size(); ++i){
    dst.data[i] = src.data[i] + dist(generator);
  }
}
float psnr( SImg<float>& gt, SImg<float>& noisy )
{
  float max_signal = 255;
  float sqr_err = 0;
  for(int i=0;i<gt.size;++i)
  {
    float diff = gt.data[i] - noisy.data[i];
    sqr_err += diff*diff;
  }
  float mse = sqr_err/gt.size;
  float psnr = 10.f*log10(max_signal*max_signal/mse);
  return psnr;
}

void read_parameters(int argc, char *argv[], Parameters &p){
 if (argc==1){
   cout<<argv[0]<<" image patch_radius window_radius sim_th maxN hard_th sigma"<<endl;
   cout<<"performing with the default settings"<<endl;
 }

 if(argc>=2) p.filename=argv[1];
 if(argc>=3) p.patch_radius=atoi(argv[2]);
 if(argc>=4) p.window_radius=atoi(argv[3]);
 if(argc>=5) p.sim_th=atof(argv[4]);
 if(argc>=6) p.maxN=atoi(argv[5]);
 if(argc>=7) p.hard_th=atof(argv[6]);
 if(argc>=8){ p.sigma=atof(argv[7]); }
 if(argc>=9){ p.noise_sigma=atof(argv[8]); }
 if(argc>=10){ p.garotte=atoi(argv[9]); }

 cout<<"Parameters:"<<endl;
 cout<<"          image: "<<p.filename.c_str()<<endl;
 cout<<"   patch_radius: "<<p.patch_radius<<endl;
 cout<<"  window_radius: "<<p.window_radius<<endl;
 cout<<"         sim_th: "<<fixed<<setprecision(3)<<p.sim_th<<endl;
 cout<<"           maxN: "<<p.maxN<<endl;
 cout<<"        hard_th: "<<fixed<<setprecision(3)<<p.hard_th<<endl;
 cout<<"          sigma: "<<fixed<<setprecision(3)<<p.sigma<<endl;
 cout<<"    noise_sigma: "<<fixed<<setprecision(3)<<p.noise_sigma<<endl;
 cout<<"        garotte: "<<fixed<<setprecision(3)<<p.garotte<<endl;
}
void compareDistances(udistvec v1, udistvec v2){
  assert(v1.size() == v2.size());
  for(unsigned k=0; k<v1.size(); ++k){
    assert(v1[k].first == v2[k].first);
    assert(v1[k].second.x == v2[k].second.x);
    assert(v1[k].second.y == v2[k].second.y);
  }
}

void dct(SImg<float> &group,
         const int z)
{
  int i;
  int rows[8][8];

  static const int  c1=1004 /* cos(pi/16) << 10 */,
        s1=200 /* sin(pi/16) */,
        c3=851 /* cos(3pi/16) << 10 */,
        s3=569 /* sin(3pi/16) << 10 */,
        r2c6=554 /* sqrt(2)*cos(6pi/16) << 10 */,
        r2s6=1337 /* sqrt(2)*sin(6pi/16) << 10 */,
        r2=181; /* sqrt(2) << 7*/

  int x0,x1,x2,x3,x4,x5,x6,x7,x8;

  /* transform rows */
  for (i=0; i<8; i++)
  {
    x0 = group(0, i, z);
    x1 = group(1, i, z);
    x2 = group(2, i, z);
    x3 = group(3, i, z);
    x4 = group(4, i, z);
    x5 = group(5, i, z);
    x6 = group(6, i, z);
    x7 = group(7, i, z);

    /* Stage 1 */
    x8=x7+x0;
    x0-=x7;
    x7=x1+x6;
    x1-=x6;
    x6=x2+x5;
    x2-=x5;
    x5=x3+x4;
    x3-=x4;

    /* Stage 2 */
    x4=x8+x5;
    x8-=x5;
    x5=x7+x6;
    x7-=x6;
    x6=c1*(x1+x2);
    x2=(-s1-c1)*x2+x6;
    x1=(s1-c1)*x1+x6;
    x6=c3*(x0+x3);
    x3=(-s3-c3)*x3+x6;
    x0=(s3-c3)*x0+x6;

    /* Stage 3 */
    x6=x4+x5;
    x4-=x5;
    x5=r2c6*(x7+x8);
    x7=(-r2s6-r2c6)*x7+x5;
    x8=(r2s6-r2c6)*x8+x5;
    x5=x0+x2;
    x0-=x2;
    x2=x3+x1;
    x3-=x1;

    /* Stage 4 and output */
    rows[i][0]=x6;
    rows[i][4]=x4;
    rows[i][2]=x8>>10;
    rows[i][6]=x7>>10;
    rows[i][7]=(x2-x5)>>10;
    rows[i][1]=(x2+x5)>>10;
    rows[i][3]=(x3*r2)>>17;
    rows[i][5]=(x0*r2)>>17;
  }

  /* transform columns */
  for (i=0; i<8; i++)
  {
    x0 = rows[0][i];
    x1 = rows[1][i];
    x2 = rows[2][i];
    x3 = rows[3][i];
    x4 = rows[4][i];
    x5 = rows[5][i];
    x6 = rows[6][i];
    x7 = rows[7][i];

    /* Stage 1 */
    x8=x7+x0;
    x0-=x7;
    x7=x1+x6;
    x1-=x6;
    x6=x2+x5;
    x2-=x5;
    x5=x3+x4;
    x3-=x4;

    /* Stage 2 */
    x4=x8+x5;
    x8-=x5;
    x5=x7+x6;
    x7-=x6;
    x6=c1*(x1+x2);
    x2=(-s1-c1)*x2+x6;
    x1=(s1-c1)*x1+x6;
    x6=c3*(x0+x3);
    x3=(-s3-c3)*x3+x6;
    x0=(s3-c3)*x0+x6;

    /* Stage 3 */
    x6=x4+x5;
    x4-=x5;
    x5=r2c6*(x7+x8);
    x7=(-r2s6-r2c6)*x7+x5;
    x8=(r2s6-r2c6)*x8+x5;
    x5=x0+x2;
    x0-=x2;
    x2=x3+x1;
    x3-=x1;

    /* Stage 4 and output */
    group(0,i,z)=(float)((x6+16)>>3);
    group(4,i,z)=(float)((x4+16)>>3);
    group(2,i,z)=(float)((x8+16384)>>13);
    group(6,i,z)=(float)((x7+16384)>>13);
    group(7,i,z)=(float)((x2-x5+16384)>>13);
    group(1,i,z)=(float)((x2+x5+16384)>>13);
    group(3,i,z)=(float)(((x3>>8)*r2+8192)>>12);
    group(5,i,z)=(float)(((x0>>8)*r2+8192)>>12);
  }
}

int main(int argc, char *argv[]){
  // Take parameters
  Parameters p;
  read_parameters(argc, argv, p);

  SImg<unsigned char> raw_image; // Original image
  cout<<"Reading image..."<<flush;
  if(! readPGM(p.filename.c_str(), raw_image) ){ cerr << "Failed to open the image.\n"; return EXIT_FAILURE;}
  cout<<"done"<<endl;

  SImg<float> original_image(raw_image.xSize, raw_image.ySize, raw_image.zSize, 0);
  SImg<float> image(raw_image.xSize, raw_image.ySize, raw_image.zSize, 0);
  simgUnsignedToFloat(original_image, raw_image);
  add_noise(image, original_image, p.noise_sigma);
  image.normalize(0, 255);
  SImg<unsigned char> asb(raw_image.xSize, raw_image.ySize, raw_image.zSize, 0);
  simgFloatToUnsigned(asb, image);
  writePGM("noisy.pgm", asb);
  CImg<float> noisy(image.data.data(), image.xSize, image.ySize, image.zSize, 1,1); noisy.display(disp1);
  cout<<"PSNR Noisy "<<psnr(original_image, image)<<endl;

  SImg<float> denoised(image.xSize, image.ySize, 0); // Denoised image
  SImg<float> weights(image.xSize, image.ySize, 0);  // Matrix with accumulated group weights
  upatches  vec_patches;  // Vector with patch center pixels ((x1,y1), (x2,y2),...)
  // Vector with cumulative sum of patches in each group e.g. (5,8,13)
  // which means fist group has 5 patches, second 3, third 5, etc...
  upatchnum num_patches;

  cout<<"Performing block matching..."<<endl;
  clock_t start = clock();
  blockMatching(image, vec_patches, num_patches, p.patch_radius, p.window_radius, p.sim_th, p.maxN);
  cout<<"Block matching took "<<(clock()-start)/(double) CLOCKS_PER_SEC<<endl;
  cout<<"done"<<endl;

  cout<<"Performing BM3D..."<<endl;
  for(unsigned i=0;i<num_patches.size();i++){
    SImg<float> group = gatherPatches(i, vec_patches, num_patches, image, p.patch_radius);
    CImg<float> ggroup(group.data.data(), group.xSize, group.ySize, group.zSize, 1,1); ggroup.display();
    for(int i_d=0; i_d<group.zSize; ++i_d){
      dct(group, i_d);
    }
    CImg<float> idct(group.data.data(), group.xSize, group.ySize, group.zSize, 1); idct.display();
    SImg<float> coeff = waveletGroupTransform(group);
    //CImg<float> test(coeff.data.data(), coeff.xSize, coeff.ySize, coeff.zSize, 1); test.display();
    float group_weight(0);
    thresholdCoeff(coeff, p, group_weight);
    //cout<<"Group weight "<<group_weight<<endl;
    //CImg<float> thc(coeff.data.data(), coeff.xSize, coeff.ySize, coeff.zSize, 1); thc.display();
    SImg<float> rec_group = inverseWaveletGroupTransform(coeff);
    //CImg<float> testg(rec_group.data.data(), rec_group.xSize, rec_group.ySize, rec_group.zSize, 1,1); testg.display();
    // assertGroups(group, rec_group);
    aggregate(i, denoised, weights, vec_patches, num_patches, rec_group, group_weight, p.patch_radius);
  }
  cout<<"done"<<endl;
  for(int i=0; i<denoised.size; ++i){
    denoised.data[i] /= weights.data[i];
  }
  cout<<"PSNR Denoised "<<psnr(original_image, denoised)<<endl;
  CImg<float> ddenoised(denoised.data.data(), denoised.xSize, denoised.ySize, denoised.zSize, 1,1);
  SImg<unsigned char> denoised_out(raw_image.xSize, raw_image.ySize, raw_image.zSize, 0);
  simgFloatToUnsigned(denoised_out, denoised);
  writePGM("denoised.pgm", denoised_out);
  //assertGroups(image, denoised);
  ddenoised.display();

  return EXIT_SUCCESS;
}
