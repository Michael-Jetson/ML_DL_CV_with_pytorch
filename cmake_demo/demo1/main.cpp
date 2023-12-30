#include<iostream>
#include "demo1.h"
using namespace std;
int main()
{
    int i=5;
    int *p=&i;
    cout<<"*p= "<<*p<<endl;
    cout<<"p= "<<p<<endl;
    cout<<"hello world"<<endl;
    student student1;
    student * p0;
    p0=&student1;
    student1.age=5;
    cout<<student1.age<<endl;
    return 0;
}