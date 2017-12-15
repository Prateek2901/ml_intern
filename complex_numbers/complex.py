from math import pow
class complex_number:
	
	def __init__(self, x, y):
		self.x = x
		self.y = y
	
	def __add__(self, next):
		return complex_number(self.x+next.x, self.y+next.y)
	
	def __sub__(self, next):
		return complex_number(self.x-next.x, self.y-next.y)
	
	def __mul__(self, next):
		return complex_number(self.x*next.x-self.y*next.y, self.x*next.y+self.y*next.x)
	
	def __truediv__(self, next):
		try: 
			return self.__mul__(complex_number(next.x, -1*next.y)).__mul__(complex_number(1.0/(next.mod().x)**2, 0))
		except ZeroDivisionError as e:
			print (e)
			return None
	def mod(self):
		return complex_number(pow(self.x**2+self.y**2, 0.5), 0)
	
	def __str__(self, precision=2):
		if self.x==int(self.x) and self.y==int(self.y):	
			return str(("%d"%(self.x))+('+' if self.y>=0 else '-'))+str(("%d")%(abs(self.y)))+'i'
		else:
			return str(("%."+"%df"%precision)%float(self.x))+('+' if self.y>=0 else '-')+str(("%."+"%df"%precision)%float(abs(self.y)))+'i'
	
	def real(self):
		return self.x

	def imag(self):
		return self.y

	def argument(self):
		return pow((pow(self.x,2)+pow(self.y,2)),.5)

	def conjugate(self):
		if self.y > 0:
			return complex_number(self.x,-1*self.y)
		else:
			return complex_number(self.x,self.y)

a,b = input().strip().split()
x,y = input().strip().split()
A = complex_number(float(a),float(b))
B = complex_number(float(x),float(y))


print (A+B)
print (A-B)
print (A*B)
print (A/B)
print (A.mod())
print (B.mod())

print (A.real())
print (B.real())

print (A.imag())
print (B.imag())

print (A.argument())
print (B.argument())

print (A.conjugate())
print (B.conjugate())

print (A)
print (B)