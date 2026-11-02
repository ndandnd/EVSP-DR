// Algorithm X with reversible sparse linked lists, independently checking a finite set of trip rows.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Matrix {
    std::vector<int> L,R,U,D,C,size;
    std::uint64_t nodes=0;
    explicit Matrix(int n) : L(n+1),R(n+1),U(n+1),D(n+1),C(n+1),size(n+1) {
        for(int c=0;c<=n;++c) {
            L[c]=c==0?n:c-1;
            R[c]=c==n?0:c+1;
            U[c]=D[c]=C[c]=c;
        }
    }
    void row(const std::vector<int>& trips) {
        int first=-1;
        for(int t:trips) {
            const int c=t+1;
            if(c<=0||c>=int(size.size())) throw std::runtime_error("trip outside domain");
            const int i=int(C.size());
            C.push_back(c);U.push_back(U[c]);D.push_back(c);U[c]=i;D[U[i]]=i;++size[c];
            if(first<0) {L.push_back(i);R.push_back(i);first=i;}
            else {L.push_back(L[first]);R.push_back(first);R[L[i]]=i;L[first]=i;}
        }
    }
    void cover(int c) {
        R[L[c]]=R[c];L[R[c]]=L[c];
        for(int i=D[c];i!=c;i=D[i]) for(int j=R[i];j!=i;j=R[j]) {
            D[U[j]]=D[j];U[D[j]]=U[j];--size[C[j]];
        }
    }
    void uncover(int c) {
        for(int i=U[c];i!=c;i=U[i]) for(int j=L[i];j!=i;j=L[j]) {
            ++size[C[j]];U[D[j]]=D[U[j]]=j;
        }
        R[L[c]]=L[R[c]]=c;
    }
    bool solve() {
        ++nodes;
        if(R[0]==0) return true;
        int c=R[0];
        for(int i=R[c];i!=0;i=R[i]) if(size[i]<size[c]) c=i;
        if(size[c]==0) return false;
        cover(c);
        for(int i=D[c];i!=c;i=D[i]) {
            for(int j=R[i];j!=i;j=R[j]) cover(C[j]);
            if(solve()) return true;
            for(int j=L[i];j!=i;j=L[j]) uncover(C[j]);
        }
        uncover(c);
        return false;
    }
};

int main(int argc, char** argv) {
    if(argc!=3) return 2;
    Matrix matrix(std::stoi(argv[1]));
    std::ifstream source(argv[2]);
    if(!source) return 2;
    std::string line;
    int rows=0;
    while(std::getline(source,line)) {
        std::vector<int> trips;
        std::istringstream stream(line);
        int t;
        while(stream>>t) trips.push_back(t);
        std::sort(trips.begin(),trips.end());
        if(trips.empty()||std::adjacent_find(trips.begin(),trips.end())!=trips.end()) return 2;
        matrix.row(trips);++rows;
    }
    const bool found=matrix.solve();
    std::cout << "{\"eligible_trip_sets\":" << rows << ",\"nodes\":" << matrix.nodes
              << ",\"found_partition\":" << (found?"true":"false") << "}\n";
}
