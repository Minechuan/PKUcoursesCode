## 数据库概论第三次作业

毛川 2300013218

### Q1：

模式为：

* `S(SNO, SNAME, CITY)`
* `P(PNO, PNAME, COLOR, PRICE)`
* `J(JNO, JNAME, CITY)`
* `SPJ(SNO, PNO, JNO, QTY)`

(1): 求向北京的工程供应了红色零件的供应商姓名

\[
T_0 = (S \bowtie_{S.SNO=SPJ.SNO} SPJ)
\bowtie_{SPJ.PNO=P.PNO} P
\bowtie_{SPJ.JNO=J.JNO} J \\
\pi_{SNAME}
\Big(
\sigma_{P.COLOR=红色 \wedge J.CITY=北京}
(T)
\Big)
\]

(2): 求同时向位于北京和天津的工程供应了零件的供应商的供应商名


\[
A = \pi_{SNO}(\sigma_{J.CITY=北京}(SPJ \bowtie_{SPJ.JNO=J.JNO} J)) \\
B = \pi_{SNO}(\sigma_{J.CITY=天津}(SPJ \bowtie_{SPJ.JNO=J.JNO} J)) \\
\pi_{SNAME}
\Big(
S \bowtie
(A\cap B)
\Big)
\]

(3): 求向和自己位于相同城市的**工程**供应零件的供应商的供应商姓名


\[ T_1 = 
\pi_{S.SNO}
\Big(
\sigma_{S.CITY=J.CITY}
\big(
(S \bowtie_{S.SNO=SPJ.SNO} SPJ)
\bowtie_{SPJ.JNO=J.JNO} J
\big)
\Big)
\]

(4): 求向和自己位于不同城市的工程供应零件的供应商的供应商号


\[
\pi_{SNO}(SPJ)- T_1
\]

(5): 求向所有位于北京的工程都供应了零件的供应商的供应商号


\[
\pi_{SNO,JNO}(SPJ)
\div
\pi_{JNO}(\sigma_{CITY=北京}(J))
\]

(6): 求价格最高的零件的零件号


\[
P_1=\rho_{P_1}(P),\qquad
P_2=\rho_{P_2}(P) \\
\pi_{P_1.PNO}(P_1)-
\pi_{P_1.PNO}
\big(
\sigma_{P_1.PRICE<P_2.PRICE}(P_1 \times P_2)
\big)
\]

### Q2:

`SC(sno, cno, grade)`

(1). 求恰好选修了 c1 和 c2 课程的学生:

\[
\left(
\pi_{sno}\big(\sigma_{cno='c1'}(SC)\big)
\;\cap\;
\pi_{sno}\big(\sigma_{cno='c2'}(SC)\big)
\right)
\;-\;
\pi_{sno}\big(\sigma_{cno\neq 'c1' \wedge cno\neq 'c2'}(SC)\big)
\]

(2). 求选修了所有 s1 同学所修课程的学生

\[
\pi_{sno,cno}(SC)
\div
\pi_{cno}\big(\sigma_{sno='s1'}(SC)\big)
\]

(3). 求其选修课程被 s1 同学所修课程完全包含的学生

\[
\pi_{sno}(SC) -
\pi_{sno}\left(
\pi_{sno,cno}(SC)-
\Big(
\pi_{sno}(SC)\times \pi_{cno}\big(\sigma_{sno='s1'}(SC)\big)
\Big)
\right)
\]

(4). 求和 s1 同学所修课程完全不同的学生

\[
\pi_{sno}(SC)-
\pi_{sno}\left(
\pi_{sno,cno}(SC)
\bowtie
\pi_{cno}\big(\sigma_{sno='s1'}(SC)\big)
\right)
\]


### Q3:

(1). 对于关系 R(A, B)，用关系代数来检验 A 是否取值唯一。

\[
E=
\sigma_{A_1=A_2 \wedge B_1\neq B_2}(R_1\times R_2)
\]
\[
E=\varnothing \iff A\text{ 在 }R(A,B)\text{ 中取值唯一}
\]


(2). 更进一步，对于关系 R(A, B, C)，用关系代数来检验 A 是否取值唯一。

\[
E=
\sigma_{A_1=A_2 \wedge (B_1\neq B_2 \vee C_1\neq C_2)}(R_1\times R_2)
\]
\[
E=\varnothing \iff A\text{ 在 }R(A,B,C)\text{ 中取值唯一}
\]

(3).

\[
左外连接=
(R \bowtie S)
\;\cup\;
\left(
\big(
R-\pi_{A,B}(R\bowtie S)
\big)
\times
\rho_{N(C)}(\{(null)\})
\right)
\]

### Q4:

(1). 求同时选修了 c1 和 c2 课程的学生
**元组关系演算：**
\[
\{\,t \mid \exists u\,\exists v\,
\big(
SC(u)\wedge SC(v)\wedge
u[cno]=c1\wedge
v[sno]=u[sno]\wedge
v[cno]=c2\wedge
t[sno]=u[sno]
\big)\,\}
\]

**域关系演算：**
\[
\{\,\langle x\rangle \mid \exists g_1\,\exists g_2\,
\big(
SC(x,c1,g_1)\wedge SC(x,c2,g_2)
\big)\,\}
\]

(2). 求选修 c1 课程且成绩比 s1 同学该门课程成绩高的学生

**元组关系演算：**
\[
\{\,t \mid \exists u\,\exists v\,
\big(
SC(u)\wedge SC(v)\wedge
t[sno]=u[sno]\wedge
u[cno]=c1\wedge
v[sno]=s1\wedge
v[cno]=c1\wedge
u[grade]>v[grade]
\big)\,\}
\]

**域关系演算：**
\[
\{\,\langle x\rangle \mid \exists g_1\,\exists g_2\,
\big(
SC(x,c1,g_1)\wedge
SC(s1,c1,g_2)\wedge
g_1>g_2
\big)\,\}
\]