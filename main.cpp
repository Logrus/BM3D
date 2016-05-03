#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <vector_types.h>
#include <vector_functions.h>
#include "CImg.h"
using namespace std;
using namespace cimg_library;

#define idx(x,y,x_size) ((x) + (y)*(x_size))
#define CLIP(minv,val,maxv) (min((maxv), max((minv),(val))))

typedef vector<unsigned char> uimg;
typedef vector<int2> upatches;
typedef vector<int> upatchnum;

CImgDisplay debug;

// Simple image class
class simg{
public:
  inline simg(): xSize(0), ySize(0), zSize(0), maxVal(0) {  };
  inline simg(const uimg &in_img, int width, int height, int channels=0, int maxVal=255): xSize(width), ySize(height), zSize(channels), maxVal(maxVal){
    data = in_img;
    size = xSize*ySize*zSize;
  };
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
    const unsigned char c_ref[] = {155};
    const unsigned char c_mat[] = {255, 0, 0};
    for(int i=start; i<Np;++i){
        pair<int2,int2> ref = getPatchBeginEnd(patches[i], k, xSize, ySize); 
        img.draw_rectangle(ref.first.x,ref.first.y,ref.second.x,ref.second.y,c_mat, 0.5);
    }
    img.display(debug);
}
void blockMatching( const simg &image, upatches &patches, upatchnum &npatches, int N, float Th, int maxN, int k ){
  int xSize = image.xSize;
  int ySize = image.ySize;
  int step=k; 
  uint ref_patch_count(0);
  int start(0);
  // Go through the image with step
  for (int j=0; j<image.ySize; j+=step)
    for (int i=0; i<image.xSize; i+=step){
    
    // The reference path is i,j
    ref_patch_count++;
    patches.push_back( make_int2(i, j) );
    npatches.push_back(1); // Allocate memory

    // Cut boundary of the window if it exceeds image size
    int wxb = max(0, i - N); // window x begin
    int wyb = max(0, j - N); // window y begin
    int wxe = min(xSize - 1, i + N); // window x end
    int wye = min(ySize - 1, j + N); // window y end 
    
    // Go through the window
    for (int wy = wyb; wy <= wye; wy++)
      for (int wx = wxb; wx <= wxe; wx++){
        float distance = dist(image, 8, make_int2(i,j), make_int2(wx,wy), xSize, ySize);
        //cout << "Distance: " << distance << endl;
        if (i!=wx && j!=wy && distance<Th && npatches[ref_patch_count-1]<=maxN ){
          patches.push_back( make_int2(wx, wy) );
          npatches[ref_patch_count-1]++;
        }
       }
    cout <<"Patch nr: "<<ref_patch_count<<", number in the group: "<<npatches[ref_patch_count-1]<<endl;
    //cout << "Start "<<start<<endl;
    drawGroup(image, patches, npatches, k, xSize, ySize, start, start+npatches[ref_patch_count-1]);
    start += npatches[ref_patch_count-1];
  }
  
}

int main(){
  simg in_image;      // Original noisy image 
  int k(4);           // Patch size
  int N(20);          // Search window
  float Th(1000.0);    // Similarity threshold for the first step
  int maxN(15);        // Maximal number of the patches in one group
  upatches patches;   // Vector 
  upatchnum npatches; // Vector 
  if(! readPGM("barbara.pgm", in_image) ){ cerr << "Failed to open the image.\n"; return EXIT_FAILURE;}
  
  blockMatching(in_image, patches, npatches, N, Th, maxN, k);
  // tranformThresholdingITransform
  // Aggregation
  writePGM("denoised.pgm", in_image);

  return EXIT_SUCCESS;
}
