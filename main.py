import numpy as np
import math
from itertools import combinations


def revers(a):  #переворот списка
    res = []
    for i in range(len(a)):
        res.append(a[len(a) - 1 - i])
    return res

def wt(a):  #расчет веса Хэмминга
    wt = 0
    for i in range(int(len(a))):
        if (a[i]):
            wt += 1
    return wt

def subset(r, m):  #формирование списка подмножеств
    el = []
    for i in range(m):
        el.append(m - 1 - i)#добавляем в список все элементы от m-1 до 0
    comb = []
    for i in range(r + 1):
        comb.append(list(j for j in combinations(el, i)))#добавляем в список все возможные комбинации
        #выбранных выше эл-тов. размеры кобинаций от 0 до 2
    res = []
    for i in range(len(comb)):
        curr = comb[i]
        for j in range(len(curr)):
            res.append(list(curr[j]))#каждую комбинацию добавляем в новый список как отдельный эл-т
    for i in range(m + 1, len(res)):
        res[i] = revers(res[i])#упорядочиваем элементы внутри каждой комбинации
    return res#получаем упорядоченный список подмножеств

def binary_list(m,n):  #формирование списка двоичных представлений
    res = []
    for i in range(n):
        j = bin(i)#число в виде двоичной строки
        j = [int(x) for x in list(j[2:len(j)])]#считываем после префикса 0b
        l = len(j)#длина считанного
        if (l < m):
            j = [0] * (m - l) + j #массив из m-l нулей  и j на конце
        res.append(revers(j))#реверс массива и добавление в список
    return res

def RMG():  #формирование матрицы кода Рида-Маллера
    sl = subset(r, m)#список подмножеств
    br = binary_list(m, n)#список двоичных представлений
    res = np.zeros((len(sl), len(br)), dtype=int)#матрица из нулей. размеры = размерам списков выше
    res[0] = np.ones(len(br), dtype=int)#первая строка матрицы единичная
    for i in range(1, len(sl)):#начиная со второй строки
        row = sl[i]# row - соответствующий эл-т спсика подмн-в
        for j in range(len(br)):
            col = br[j]#col - j-ое двоичное представление
            zero_num = 0
            for a in range(len(row)):
                if (col[row[a]] == 0):
                    zero_num = zero_num + 1
            if (zero_num == len(row)):#все разряды, входящие в подмн-во соотв. столбца, равны нулю
                res[i][j] = 1#эл-т матрицы =1
    return res

def Encode(a, G):  #кодирование последовательности
    return (a @ G) % 2

def comp_subset(r,m):  #формирование списка комплементарных множеств
    elements = []
    for i in range(m):
        elements.append(i)#элементы от 0 до m-1 вкл.
    sl = subset(r,m)#список обычных подмн-в
    res = []
    for i in range(len(sl)):
        curr = []#ля каждого подмн-ва
        for j in range(len(elements)):
            if elements[j] not in sl[i]:#если эл-та нет в обычном подмн-ве,
                curr.append(elements[j])#эл-т есть в комплементарном подмн-ве
        res.append(curr)
    return res

def base_vector(i):  #формирование базового вектора для i-го компл. подмножества - аналогично RMG()
    cs = comp_subset(r,m)
    br = binary_list(m,n)
    res = np.zeros((len(cs), len(br)), dtype=int)
    res[0] = np.ones(len(br), dtype=int)
    for j in range(1, len(cs)):
        row = cs[j]
        for k in range(len(br)):
            col = br[k]
            zero_num = 0
            for a in range(len(row)):
                if (col[row[a]] == 0):
                    zero_num = zero_num + 1
            if (zero_num == len(row)):
                res[j][k] = 1
    return res[i].tolist()#но возвращаем i-ую строку матрицы

def shifts(i):  #формирование двоичного представления сдвигов для i-го подмножества
    br = binary_list(m, n)#список двоичных представлений
    sl = subset(r, m)[i]#i-тое подмн-во
    res = []
    for j in range(len(br)):
        curr = br[j]#j-тое двоичное пр-ие
        check = 0
        for k in range(len(sl)):
            if (curr[sl[k]] != 0):
                check = 1
        if (check == 0):#если разряды двоичного представления, входящие в i-тое подмн-во,
                        # #нулевые, такой сдвиг допустим
            res.append(curr)
    return res


def check_vector(i):  #формирование проверочных векторов для i-го подмножества
    Shifts = shifts(i)#сдвиги для i-того подмн-ва
    for j in range(len(Shifts)):
        Shifts[j] = revers(Shifts[j])#переворот сдвига
        Shifts[j] = ''.join(str(e) for e in Shifts[j])#как строка
        Shifts[j] = int(Shifts[j], base=2)#переводим в int
    res = []
    b = base_vector(i)#базовый вектор для i-го компл. подмножества
    for k in range(len(Shifts)):
        curr = b.copy()
        curr = curr[-Shifts[k]:] + curr[:-Shifts[k]]#сдвигаем базовый вектор влево
                                        # на каждое число бит из Shifts по очереди
        res.append(curr)
    return res

def Decode(w,k):  #декодирование последовательности
    sl = subset(r,m)#список подмножеств
    res = [0] * k #массив из k 0 - для исходного сообщения
    G = RMG()#матрица
    a = r
    for i in range(int(len(sl)) - 1, -1, -1):
        if (len(sl[i]) == a or i == 0):#для каждого подмн-ва |J|=r
            v = check_vector(i)#проверочные вектора
            scalars = []
            for j in range(len(v)):
                curr = v[j]
                sum = 0
                for l in range(len(curr)):
                    sum = (sum + curr[l] * w[l]) % 2#скалярное пр-ие w(i) и проверочных векторов по м.2
                scalars.append(sum)
            ones = 0
            for s in range(len(scalars)):
                if (scalars[s] == 1):
                    ones = ones + 1#сумма 1 в скалярном произведении
            if (ones == len(scalars) / 2):#число 1 и 0 совпадает
                print("Ошибка не может исправлена!")
                return 0
            if (ones > 2 ** (m - a - 1)):#если число 1 > числа 0,
                res[i] = 1#соотв разряд исходного сообщения = 1
                w = (w - G[i]) % 2
        else:
            a = a - 1
        if (wt(w) <= 2 ** (m - r - 1) - 1):#если wt(w(i-1)) не превышает указанного размера, остановить алгоритм
                break
    return res


print("ЧАСТЬ 1\n4.1")
while True:
    r = int(input("Введите r от 1 до 3: "))
    if not 1 <= (r) <= 3:
        print("Попробуйте снова")
    else:
        print("r =", r)
        break
while True:
    m = int(input("Введите m от 1 до 4: "))
    if not 1 <= (m) <= 4:
        print("Попробуйте снова")
    elif m<r:
        print("Попробуйте снова")
    else:
        print("m =", m)
        break

#расчет размерностей
k = 0
for i in range (r + 1):
    k = k + int(math.factorial(m)/(math.factorial(i)*math.factorial(m-i)))
n = 2**m
print("n =", n,"\nk =", k )
print('Список подмножеств:')
print(subset(r, m))
print('Список двоичных представлений:')
Bin = binary_list(m, n)
for i in range(len(Bin)):
    print(Bin[i])

print("Матрица кода Рида-Маллера:")
G = RMG()
print(G)
print("4.2")
print('Список комплементарных подмножеств:')
print(comp_subset(r, m))
while True:
    i = int(input("Введите i от 1 до k\n"))
    if not 0 <= (i) <= k:
        print("Попробуйте снова")
    else:
        print("i =", i)
        break
print("Базовый вектор для", i, "комплементарного подмножества")
print(base_vector(i))
print("Сдвиги  для ", i, "подмножества в двоичном виде")
print(shifts(i))
print("Проверочные векторы для ", i, "подмножества в двоичном виде")
Check = check_vector(i)
for j in range(len(Check)):
    print(Check[j])
print("Проверочные векторы для ", i, "подмножества в двоичном виде")

mas2 = []
print("Введите", k, "символов")
mas2 = [int(input()) for i in range(k)]#ввод сообщения

Encod = Encode(mas2, G)
print("Исходное сообщение:\n",mas2, "\nЗакодированное сообщение:\n",Encod)

#кол-во ошибок от 1 до 4
while True:
    count_er = int(input("Количество ошибок?: "))
    if not 1 <= (count_er) <= 4:
        print("Попробуйте снова")
    else:
        print(count_er)
        break
mas_er2=[]
z=0
#последовательно вносим ошибки
while z < count_er:
    i = int(input("В какой бит внести ошибку?: "))
    if not 0 <= (i) < n: #не выходим за границы сообщения
        print("Число не в диапазоне, попробуйте снова")
        i = int(input("В какой бит внести ошибку?: "))
    elif i in mas_er2:
        print("Число ранее было задано,попробуйте снова")#е меняем один и тот же бит несколько раз
        i = int(input("В какой бит внести ошибку?: "))
    else:
        print("i =", i)
        mas_er2.append(i)#запоминаем номер бита для проверки
        Encod[i] = not Encod[i]#еняем бит
        print("Cлово с ошибкой в бите",i,":", Encod)
    z+=1
Decod = Decode(Encod,k)
print("Декодированное сообщение после внесения ошибок:\n",Decod)










