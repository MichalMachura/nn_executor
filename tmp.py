import ast,torch

def fcnnn(*args,**kwargs):
    return args, kwargs

o = torch.nn.Conv2d(3,4,3)
print(o)
s = str(o)
print(s)
d = s[s.find('('):]
print(d)
py_txt = o.__class__.__module__+'.'+o.__class__.__name__+d
print(py_txt)


v:torch.nn.Module
exec('v='+py_txt)
# v = ast.parse(py_txt)
print(v)


print()