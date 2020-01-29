import re

pattern="((\d)\?[a-z0-9]*?\?[a-z0-9]*?\?[a-z0-9]*?(\d))"
stirng="arrb6???4xxbl5???eee5"
groups=re.findall(pattern,stirng)

def validate(groups):
    for g in groups:
        sum=int(g[1])+int(g[2])
        print(sum)
        if sum == 10:
            return True
    return False


value=validate(groups)
print(value)