#!/usr/bin/python
print 'hello world'
a=[1,4,5,8];
a=a+[9,11];
box = {'fruits': ['apple','orange'], 'money': 1993, 'name': 'obama'};
 
msg='hello world';
for c in msg:
  print c;
b=[x*x for x in a if x%2==0];
print b;

print box['money']

from numpy import *

c=random.rand(4,4);

print c;
for d in c:
    print d;
print c[1][2];

g=mat(random.rand(2,4));
e=mat(random.rand(4,3));

f=g*e;
print f;
print f.I;
print f*f.I;
print eye(4);
