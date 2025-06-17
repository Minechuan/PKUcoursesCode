/*
遇到的问题：如果使用多态，即将Lion指针赋值给Warrior指针数组，那么如何获得Lion的忠实度，或者使Lion逃跑
应该不会销毁原有的成员变量，实际上无法调用；
为了解决这样的问题：把那些会再次使用的东西放在虚函数里，可以作用到；
*/
#include<iostream>
#include<string>
#include<iomanip>
#include<algorithm>
using namespace std;
int IcemanblueNum = 0;//把所有静态变量设置为零
int IcemanredNum = 0;
int NinjablueNum = 0;
int NinjaredNum = 0;
int LionblueNum = 0;
int LionredNum = 0;
int DragonblueNum = 0;
int DragonredNum = 0;
int WolfblueNum = 0;
int WolfredNum = 0;
//在城市中Warroir以指针形式储存
//未封装部分
void NewCase() {
	IcemanblueNum = 0;//把所有静态变量设置为零
	IcemanredNum = 0;
	NinjablueNum = 0;
	NinjaredNum = 0;
	LionblueNum = 0;
	LionredNum = 0;
	DragonblueNum = 0;
	DragonredNum = 0;
	WolfblueNum = 0;
	WolfredNum = 0;
}
class Arsenal;
class Weapon;
class Warrior;
class Sword;
class Bomb;
class Arrow;
class City;
class control;
/*********************************************
******			   全局变量				******
*********************************************/
int debUg = 0;
int BloodArray[5] = { 0,0,0,0,0 };
int StrengthArray[5] = { 0,0,0,0,0 };
string WeaponArray[3] = { "sword","bomb","arrow"};
Warrior* redHeadquarterW=NULL;
Warrior* blueHeadquarterW=NULL;
int LoyalityDecrease = 0;//每轮狮子如果被打败减少的忠诚值
int TotalTime = 0;
City * FirstCity=NULL;
City * LastCity=NULL;
//静态变量无法在外部函数被初始化，所以我设置静态变量的目的是预先假设其值基本不变
/*用基类指针数组存放指向各种派生类对象的指
针，然后遍历该数组，就能对各个派生类对象
做各种操作，是很常用的做法*/
/********************
**		控制		  ***
********************/
class control {//希望把创建的函数写在control里面；
public://用来存放全局变量
	static int cityNum;//城市的数量；
	static int time;//二者共用一个time
	static void getBloodArray();
	static void getStrengthArray();
	static int minBlood;//五种武士所需要的最小晶元；
	//static bool color;//轮到哪一方
	static void Timeprint();//可以使用静态变量
	inline static void timeAdd(int increase) {
		time += increase;
	}//在得到一个新的Case时需要将已经设置的静态变量重新初始化；
	~control() {};
};
void control::getBloodArray() {
	int temp = 0;
	for (int i = 0; i < 5; i++) {
		cin >> temp;
		BloodArray[i] = temp;
	}
}
void control::getStrengthArray() {
	int temp = 0;
	for (int i = 0; i < 5; i++) {
		cin >> temp;
		StrengthArray[i] = temp;
	}
}
void control::Timeprint() {
	int hours = control::time / 60;
	if (hours < 10)
		cout << "00";
	else if (hours >= 10 && hours < 100)
		cout << "0";
	cout <<hours << ":" <<setw(2)<<setfill('0') <<time % 60 << ' ';
}
/************************
*		武器				*
*		武器库			*
************************/
class Weapon {
public:
	int order;//编号
	double ForceRate;
	Weapon(int order_, double ForceRate_) {
		order = order_;
		ForceRate = ForceRate_;
	}
	virtual void UsePower(Warrior* Attacker, Warrior* Defender) {}//纯虚函数,攻击者和被攻击者
	virtual int getUseTime() {
		return 100000;
	}
};
class Sword :public Weapon {
public:
	Sword() :Weapon(0, 0.2) {}
	virtual void UsePower(Warrior* Attacker, Warrior* Defender);//传入两个指向武士的指针,使用方法在武士类的定义之后
};
class Bomb :public Weapon {
public:
	int UseTimes;
	Bomb() :UseTimes(1), Weapon(1, 0.4) {}
	virtual void UsePower(Warrior* Attacker, Warrior* Defender);
	virtual int getUseTime() {
		return UseTimes;
	}
};
class Arrow :public Weapon {
public:
	int UseTimes;
	Arrow() :UseTimes(2), Weapon(2, 0.3) {}
	virtual void UsePower(Warrior* Attacker, Warrior* Defender);
	virtual int getUseTime() {
		return UseTimes;
	}
};
//#####################################################################
class Arsenal {//其实也可以使用STL中的优先队列
public:
	//可以用三个长度为十的指针数组来存放现有的武器；
	Weapon* myWeapon[10];//十个指向Weapon的指针，需要在Weapon和其派生类中实现多态；
	void SortByOrder();
	void DeleteUsedOne();
	void AddNewWeapon(Weapon* NewWeapon);
	void Sort_When_Loss_Or_Steal();
	int WeaponNumber();
	void PushFront();
	Arsenal() { for (int i = 0; i < 10; i++) { myWeapon[i] = NULL; } }//初始化
};
int Arsenal::WeaponNumber() {//当前拥有的武器数量
	int num = 0;//中间不会有空格
	for (int i = 0; i < 10; i++) {
		if (myWeapon[i] != NULL) {
			num++;
		}
		else {
			return num;
		}
	}
	return num;
}
bool cmp(Weapon* a,Weapon* b) {//纯虚函数不能使用cmp
	if (a->order != b->order||a->order!=2) {
		return a->order < b->order;
	}
	else{
		return a->getUseTime() < b->getUseTime();//用过的弓箭排在前面
	}
}
bool anotherCMP(Weapon* a, Weapon* b) {//纯虚函数不能使用cmp
	if (a->order != b->order || a->order != 2) {
		return a->order < b->order;
	}
	else {
		return a->getUseTime() > b->getUseTime();//用过的弓箭排在前面
	}
}
void Arsenal::SortByOrder() {//依照序号给已经抢到的武器排序
	int lastNum = WeaponNumber();
	if (lastNum > 1) {
		sort(myWeapon, myWeapon+lastNum, cmp);
	}
}
void Arsenal::Sort_When_Loss_Or_Steal() {
	int lastNum = WeaponNumber();
	if (lastNum > 1)
		sort(myWeapon, myWeapon+lastNum,anotherCMP);
}
void Arsenal::AddNewWeapon(Weapon * newWeapon) {//加入一个新获取的武器
	int lastNum = WeaponNumber();
	myWeapon[lastNum] = newWeapon;
	SortByOrder();
}
void Arsenal::DeleteUsedOne() {//删除一个已经使用完的武器,设置为空指针
	int lastNum = WeaponNumber();
	for (int i = 0; i < lastNum; i++) {
		if (myWeapon[i]->getUseTime() == 0) {
			myWeapon[i] = NULL;
		}
	}//在战争过程中调用
}
void Arsenal::PushFront(){
	//将不空的指针向前移动紧凑
	int leftNULL = 0;
	for (int i = 0; i < 10; i++) {
		if (myWeapon[i] != NULL&&i>leftNULL) {
			myWeapon[leftNULL] = myWeapon[i];
			myWeapon[i] = NULL;
			leftNULL++;
		}
		else if (myWeapon[i] != NULL && i == leftNULL) {
			leftNULL++;
		}
	}
	SortByOrder();
}
Weapon* nweapon(int weapon_order) {//分配武器时调用
	Weapon* newWeapon;
	if (weapon_order == 0) {
		newWeapon = new Sword;
	}
	else if (weapon_order == 1)
		newWeapon = new Bomb;
	else
		newWeapon = new Arrow;
	return newWeapon;
}
void PrintOwnerOrder(int gen, int color) {//获取并更改每一方该种武士的数量
	switch (gen)
	{
	case 0: cout << (color == 0 ? ++DragonredNum : ++DragonblueNum); break;
	case 1:cout << (color == 0 ? ++NinjaredNum : ++NinjablueNum); break;
	case 2: cout << (color == 0 ? ++IcemanredNum : ++IcemanblueNum); break;
	case 3:cout << (color == 0 ? ++LionredNum : ++LionblueNum); break;
	case 4:cout << (color == 0 ? ++WolfredNum : ++WolfblueNum); break;
	}
}
/***************************************
******			  武士				****
***************************************/
class Warrior {
public:
	int blood;//同时也是制造这个需要消耗的晶元数
	int strength;//攻击力
	int color;//在己方阵营中的代号
	int order;//所属阵营
	string ItsName;//用来鉴定武士的类别
	Arsenal OwnArsenal;//该武士的武器库
	Warrior(int bloodIn, int strength_, int color_, int order_, string name, int gen);
	void ReporttWeaponAndBlood();//报告武器情况
	//void  Attack(Warrior* PWarrior);
	//void Hurt(const int& lossblood);
	//虚函数，有差异性
	virtual void GainWeaponWhenBorn(int order) {};//虚函数
	virtual void TakeActions1() {}//狮子减少Loyality
	virtual bool TakeActions2() { return 0; };//武士欢呼或者狮子逃跑//能够调用到派生类中个性化的成员函数；需要在派生类中也定义虚函数连接；
	virtual void stealWeapon(Warrior* Enemy,int cityorder) {};//实际上只针对wolf
	//virtual void FightBack() {};
};
void printWhenBornAWarrior(int blood, int strength, int color, int order, string name, int gen) {
	//cout << order << endl; can get right order; 
	cout << (color == 0 ? "red" : "blue") << " " << name << " " << order << " born" << endl;;
	//构造一个武士，赋予一个ID；
}
Warrior::Warrior(int bloodIn, int strength_, int color_, int order_, string name, int gen) {
	//构造函数不能被子累继承,在构造的时候将需要输出的内容输出；
	blood = bloodIn;
	strength = strength_;
	color = color_;
	order = order_;
	ItsName = name;
	printWhenBornAWarrior(bloodIn, strength_, color_, order_, name, gen);
};
void Warrior::ReporttWeaponAndBlood() {//报告武士的情况
	int NumSword = 0, NumBomb = 0, NumArrow = 0;
	for(int i = 0; i < 10; i++){
		if (OwnArsenal.myWeapon[i]) {//如果有武器则输出答应

		}
		else {
			break;
		}
	}
	cout << NumSword << NumBomb << NumArrow;
	cout << "with " << blood << " elements and force " << strength << endl;;
}
//刚出生得到武器,并且将它们加入到武器库中
//void Warrior::Attack(Warrior * PWarrior) {
	//攻击
//}
//void Warrior::Hurt(const int& lossblood) {
	//损失生命值
//}
void Sword::UsePower(Warrior* Attacker, Warrior* Defender) {
	Defender->blood -= (Attacker->strength)*2/10;//需要取整
}
void Bomb::UsePower(Warrior* Attacker, Warrior* Defender) {
	int lossEnemy = (Attacker->strength)*4/10;
	Defender->blood -= lossEnemy;//需要取整
	if (Attacker->ItsName != "ninja") {
		Attacker->blood -= (lossEnemy*5/10);
	}
	UseTimes = 0;
}
void Arrow::UsePower(Warrior* Attacker, Warrior* Defender) {
	Defender->blood -= (Attacker->strength)*3/10;//需要取整
	UseTimes--;
}
//dragon 、ninja、iceman、lion、wolf 为生成数组中对应的顺序；
// 0        1      2      3      4
// 
//需要注意，在函数中构造的变量不能在函数消亡时析构！
//那么现有的变量应该如何维护，既然武士有一个编号，那么可以根据编号存放在一个可变数组中；即以Warrior为元素的数组
//确定到底是哪一方进行建造，使用一个 bool 类型 color，如果为 0，那么红方建造，反之 1 蓝方建造； 
//		 ############
//########	Lion	#######//
//		 ############
class Lion :public Warrior {
private:
	static const int gen;
public:
	int Loyalty = 0;
	static const string name;
	Lion(int blood, int strength, int color, int order, int TheForce) : Warrior(blood, strength, color, order, this->name,this->gen),Loyalty(TheForce){
		cout << "Its loyalty is " << Loyalty << endl;
		GainWeaponWhenBorn(order);
	}
	virtual void GainWeaponWhenBorn(int order);
	virtual void TakeActions1();//降低忠实值
	virtual bool TakeActions2();//决定是否欢呼
};
const int Lion::gen = 3;
const string Lion::name = "lion";
void Lion::GainWeaponWhenBorn(int order) {
	int weapon_order = order % 3;
	OwnArsenal.myWeapon[0] = nweapon(weapon_order);
}
void Lion::TakeActions1() {
	Loyalty -= LoyalityDecrease;
}
bool Lion::TakeActions2() {//决定删除指针就return true
	if (Loyalty <= 0) {
		return true;
	}
	return false;
}
//		 ############
//########	Wolf	#######//
//		 ############
class Wolf :public Warrior {
private:
	static const int gen;
	static const string name;
public:
	Wolf(int blood, int strength, int color, int order) : Warrior(blood, strength , color, order, this->name,this->gen) {}
	virtual void stealWeapon(Warrior* Enemy,int cityorder);
};
const int Wolf::gen = 4;
const string Wolf::name = "wolf";
//		 ############
//########	Dragon	#######//
//		 ############
class Dragon :public Warrior {
private:
	static const int gen;
	static const string name;
	double Morale;
public:
	Dragon(int blood, int strength, int color, int order, int TheForce) : Warrior(blood, strength, color, order, this->name, this->gen),
		Morale(1.0 * TheForce / blood) {
		GainWeaponWhenBorn(order);
	}//cout << ",and it's morale is " << fixed << setprecision(2) << Morale << endl;
	virtual void GainWeaponWhenBorn(int order);
	virtual void TakeActions1();//决定是否欢呼
};
const int Dragon::gen=0;
const string Dragon::name = "dragon";
void Dragon::GainWeaponWhenBorn(int order) {
	int weapon_order = order % 3;
	OwnArsenal.myWeapon[0] = nweapon(weapon_order);
}
void Dragon::TakeActions1() {
	control::Timeprint();
	cout << (color == 0 ? "red " : "blue ") << ItsName << " " << order << " yelled in city ";
}

//		 ############
//########  Iceman	#######//
//		 ############
class Iceman :public Warrior {
private:
	static const int gen;
	static const string name;
public:
	Iceman(int blood, int strength, int color, int order) : Warrior(blood, strength, color, order, Iceman::name,this->gen) { GainWeaponWhenBorn(order); }
	virtual void GainWeaponWhenBorn(int order);
	void TakeActions1() {
		blood -= (blood / 10);
	}
};
const string Iceman::name = "iceman";
const int Iceman::gen = 2;//对应的编号
void Iceman::GainWeaponWhenBorn(int order) {
	int weapon_order = order % 3;
	OwnArsenal.myWeapon[0] = nweapon(weapon_order);
}

//		 ############
//########	Ninja	#######//
//		 ############
class Ninja :public Warrior {
private:
	static const int gen;
	static const string name;
public:
	Ninja(int blood, int strength, int color, int order) : Warrior(blood, strength, color, order, this->name,this->gen) {
		GainWeaponWhenBorn(order);
	}
	virtual void GainWeaponWhenBorn(int order);
};
const int Ninja::gen = 1;
const string Ninja::name = "ninja";

void Ninja::GainWeaponWhenBorn(int order) {
	int weapon_order = order % 3;
	OwnArsenal.myWeapon[0] = nweapon(weapon_order);
	weapon_order = (order + 1) % 3;
	OwnArsenal.myWeapon[1] = nweapon(weapon_order);
	OwnArsenal.SortByOrder();//给武器用编号排序
}
//明确 “is a ” 和 “has a ”的关系，确定成员应该放在哪个类中；
/*******************************
***	    司令部Headquarter	****
*******************************/
class command {//一共只需要两个对象！
public:
	int color;//表示为哪一方的司令部
	int TheForce=0;//目前的晶元数量
	int BornArray[5];//产生顺序
	int TotalNumCreated=0;//目前两个司令部累计创建的士兵数量，用来给每个阵营创造的武士编号
	int buildTurn = 0;//目前按照顺序生产到第几个
	command(int ori, int colorIn);
	bool canBuild();//判断是否能生产
	//int findWhich() ;//按照顺序查看，能创造哪个就返回那个对象！希望是可以的
	void Build();//确定能生产之后，决定生产那一个
	
};
command::command(int ori, int colorIn) {
	TheForce = ori;//构造函数，初始化某个司令部的晶元数量；
	color = colorIn;
	if (color == 0) {//construct red
		BornArray[0] = 2;
		BornArray[1] = 3;
		BornArray[2] = 4;
		BornArray[3] = 1;
		BornArray[4] = 0;
	}
	else {//blue
		BornArray[0] = 3;
		BornArray[1] = 0;
		BornArray[2] = 1;
		BornArray[3] = 2;
		BornArray[4] = 4;
	}
}
bool command::canBuild() {//此时不用通知停止制造武士
	//如果司令部中的生命元不足以制造某本该造的武士，那就从此停止制造武士。
	if (TheForce < BloodArray[BornArray[buildTurn%5]]) {//theforce 是command的对象；
		return false;
	}
	else {
		return true;
	}
}

void command::Build() {//可以到时候将返回值改变为 Warrior ;
	//并且确定需要build哪个
	//Wolf(cur.BloodArray[4],)
	control::Timeprint();
	int gen = BornArray[buildTurn%5];//确定能建造该武士；
	buildTurn++;
	TotalNumCreated++;
	switch (gen) {//分编号讨论
	case 0: {TheForce -= BloodArray[0]; Dragon* NewDragon = new Dragon(BloodArray[0],StrengthArray[0], color, TotalNumCreated, TheForce);
		(color == 0 ? redHeadquarterW = NewDragon : blueHeadquarterW = NewDragon);
		break; }
	case 1: {TheForce -= BloodArray[1]; Ninja* NewNinja = new Ninja(BloodArray[1], StrengthArray[1], color, TotalNumCreated);
		(color == 0 ? redHeadquarterW = NewNinja : blueHeadquarterW = NewNinja);
		break; }
	case 2: {TheForce -= BloodArray[2]; Iceman* NewIceman = new Iceman(BloodArray[2], StrengthArray[2], color, TotalNumCreated);
		(color == 0 ? redHeadquarterW = NewIceman : blueHeadquarterW = NewIceman);
		break; }
	case 3: {TheForce -= BloodArray[3]; Lion* NewLion = new Lion(BloodArray[3],StrengthArray[3], color, TotalNumCreated, TheForce);
		(color == 0 ? redHeadquarterW = NewLion : blueHeadquarterW = NewLion);
		break; }
	case 4: {TheForce -= BloodArray[4]; Wolf* NewWolf = new Wolf(BloodArray[4],StrengthArray[4], color, TotalNumCreated);
		(color == 0 ? redHeadquarterW = NewWolf : blueHeadquarterW = NewWolf); }
	}//把新建的武士传递给两个全局指针
}
/******************************************************
*** red   city    city   city…………………  city  blue ******
***			1      2      3             n         *****
***			WarriorInit[0]   WarriorInIt[1]       *****
******************************************************/
class City {
public:
	int order;//城市的编号
	int flag;//red: 0; blue: 1; None: -1
	enum flag {red = 0,blue=1,noflag=-1,};
	//int TheForceNum;//城市中的晶元数量
	Warrior *WarriorInIt[2];//0~4代表武士种类，用于方便的在五个指针数组中查找，若为-1则为空，因为一个城市中只能有双方各一个武士；
	City* AheadCity;
	City* LatterCity;
	//其中WarriorInIt[0]为红方的，WarriorInIt[1]为蓝方的
	City(int order_) {
		order = order_;
		flag = noflag;
		AheadCity = NULL;
		LatterCity = NULL;
		WarriorInIt[0] = NULL;
		WarriorInIt[1] = NULL;
	}
	City &operator = (const City  city);
	void CityLionRun();
	void CityWolfSteal();
	void CityDragonYell();
	void War_and_Result();
	int FindAttacker();
	int CityBeginWar(Warrior *Attacker,Warrior* Defender);//0 red win; 1 blue win; -1 平局
	int CityJudgeResult();
	void printResult(int result);
};
City& City::operator=(const City city) {
	order = city.order;
	flag = city.flag;
	AheadCity = city.AheadCity;
	LatterCity = city.LatterCity;
	return *this;
}
City* const CitiesBuild(const int& cityNum) {//得到一个双向链表
	City* City1 = new City(1);
	City1->LatterCity = new City(2);
	City1->LatterCity->AheadCity = City1;
	City* const retCity = City1;
	City1 = City1->LatterCity;
	for (int i = 3; i <= cityNum; i++) {
		City1->LatterCity = new City(i);
		City1->LatterCity->AheadCity = City1;
		City1 = City1->LatterCity;
	}
	return retCity;
}
/*****************************
***			进程函数			**
*****************************/
//在大局中，不同函数实现每一项进程
void WarriorsBorn(command * red,command * blue) {//武士在
	if (red->canBuild()) {
		red->Build();
	}
	if (blue->canBuild()) {
		blue->Build();
	}
	return;
}
//######################################################
void printRun(Warrior * W) {
	//001:05 blue lion 1 ran away
	control::Timeprint();
	cout << (W->color == 0 ? "red " : "blue ") << W->ItsName << " " << W->order << " ran away"<<endl;
}
void City::CityLionRun() {
	for (int i = 0; i < 2; i++) {
		if (WarriorInIt[i]!=NULL&&WarriorInIt[i]->ItsName == "lion")
			if(WarriorInIt[i]->TakeActions2()) {
				printRun(WarriorInIt[i]);
				WarriorInIt[i] = NULL;
			}
	}
}
void LionRunAway() {
	//但是已经到达敌人司令部的lion不会逃跑。lion在己方司令部可能逃跑。
	if (redHeadquarterW!=NULL&& redHeadquarterW->ItsName == "lion"&&redHeadquarterW->color==0) {
		if (redHeadquarterW->TakeActions2()) {
			printRun(redHeadquarterW);
			redHeadquarterW = NULL;
		}//这种情况对应刚降生就逃跑
	}
	City* checkIf_lion_run = FirstCity;
	for (int i = 1; i < control::cityNum; i++) {
		checkIf_lion_run->CityLionRun();
		checkIf_lion_run = checkIf_lion_run->LatterCity;
	}
	checkIf_lion_run->CityLionRun();
	if (blueHeadquarterW!=NULL && blueHeadquarterW->ItsName == "lion"&&blueHeadquarterW->color==1) {
		if (blueHeadquarterW->TakeActions2()) {
			printRun(blueHeadquarterW);
			blueHeadquarterW = NULL;
		}
	}
}
//######################################################
Warrior* redHenemy;
Warrior* blueHenemy;//敌军士兵会占领这个司令部

void printReach(Warrior* winW) {
	//001:10 red iceman 1 reached blue headquarter with 20 elements and force 30
	control::Timeprint();
	cout<< (winW->color == 0 ? "red " : "blue ") <<winW->ItsName <<" " << winW->order << " reached " << (winW->color == 0 ? "blue" : "red") <<
		" headquarter with " << winW->blood << " elements and force " << winW->strength << endl;
}
void printConquer(int color) {
	control::Timeprint();
	cout << (color == 0 ? "red" : "blue") << " headquarter was taken" << endl;
}
bool WarriorsReachEnemyCommand(int color) {//输出顺序可能有问题
	bool GameOver = 0;
	if (color==0&&redHenemy != NULL && redHenemy->color == 1) {
		printReach(redHenemy);
		printConquer(0);
		GameOver = 1;
	}
	if (debUg == 2) {
		int p = 3;
	}
	if (color==1&&blueHenemy != NULL && blueHenemy->color == 0) {
		GameOver = 1;
		printReach(blueHenemy);
		printConquer(1);
	}
	return GameOver;
}
void LetThemMove() {
	//cout << "First city = " << FirstCity->order << endl;
	City* cur_city = FirstCity;
	if (cur_city->WarriorInIt[1] != NULL) {
		if (cur_city->WarriorInIt[1]->ItsName == "iceman") {
			cur_city->WarriorInIt[1]->TakeActions1();
		}
		redHenemy = cur_city->WarriorInIt[1];
		cur_city->WarriorInIt[1] = NULL;//抵达敌方司令部的狮子不会逃跑
	}
	for (int i = 1; i <control::cityNum; i++) {
		cur_city = cur_city->LatterCity;
		if (cur_city->WarriorInIt[1] != NULL) {
			if (cur_city->WarriorInIt[1]->ItsName == "lion"|| cur_city->WarriorInIt[1]->ItsName == "iceman") {
				cur_city->WarriorInIt[1]->TakeActions1();
			}
			cur_city->AheadCity->WarriorInIt[1] = cur_city->WarriorInIt[1];
			cur_city->WarriorInIt[1] = NULL;
		}
	}//已经到达最后一座城市
	if (blueHeadquarterW != NULL) {
		if (blueHeadquarterW->ItsName == "lion" || blueHeadquarterW->ItsName == "iceman") {
			blueHeadquarterW->TakeActions1();
		}
		cur_city->WarriorInIt[1] = blueHeadquarterW;
		
		blueHeadquarterW = NULL;
	}
	//蓝方移动完毕
	if (cur_city->WarriorInIt[0] != NULL) {
		if (cur_city->WarriorInIt[0]->ItsName == "iceman") {
			cur_city->WarriorInIt[0]->TakeActions1();
		}
		blueHenemy= cur_city->WarriorInIt[0];
		cur_city->WarriorInIt[0] = NULL;//红方最前的武士移动至蓝方司令部
	}
	for (int i = 1; i < control::cityNum; i++) {
		cur_city = cur_city->AheadCity;
		if (cur_city->WarriorInIt[0] != NULL) {
			if (cur_city->WarriorInIt[0]->ItsName == "lion" || cur_city->WarriorInIt[0]->ItsName == "iceman") {
				cur_city->WarriorInIt[0]->TakeActions1();
			}
			cur_city->LatterCity->WarriorInIt[0] = cur_city->WarriorInIt[0];
			cur_city->WarriorInIt[0] = NULL;
		}
	}
	if (redHeadquarterW != NULL) {
		if (redHeadquarterW->ItsName == "lion"||redHeadquarterW->ItsName=="iceman") {
			redHeadquarterW->TakeActions1();
		}
		cur_city->WarriorInIt[0] = redHeadquarterW;
		//cout << cur_city->WarriorInIt[0]->ItsName <<" "<<cur_city->WarriorInIt[1]->ItsName<< endl;
		redHeadquarterW = NULL;
	}
	//至此武士已经都已经前进到
}
//000:10 blue lion 1 marched to city 1 with 10 elements and force 5
void printMarchCity(Warrior * W,int Cityorder) {
	control::Timeprint();
	cout << (W->color == 0 ? "red " : "blue ") << W->ItsName<<" "<<W->order << " marched to city " << Cityorder << " with " << W->blood
		<< " elements and force " << W->strength << endl;
}
int report_Move_result() {
	City* cur_city = FirstCity;
	int If_Gameover1 = WarriorsReachEnemyCommand(0);
	for (int ords = 0; ords < 2; ords++) {
		if (cur_city->WarriorInIt[ords] != NULL) {
			printMarchCity(cur_city->WarriorInIt[ords], cur_city->order);
		}
	}
	for (int i = 1; i < control::cityNum; i++) {
		cur_city = cur_city->LatterCity;
		for (int ords = 0; ords < 2; ords++) {
			if (cur_city->WarriorInIt[ords] != NULL) {
				printMarchCity(cur_city->WarriorInIt[ords], cur_city->order);
			}
		}
	}//已经到达最后一座城市

	int If_Gameover2= WarriorsReachEnemyCommand(1);//蓝方司令部是否被占领
	return If_Gameover1|If_Gameover2;//只要有一方被占领
}
int WarriorsMoveForward() {
	LetThemMove();//使每个武士前进
	int ifg=report_Move_result();//前进之后按照顺序报告结果
	return ifg;//int 判断游戏是否结束
}

//########################################################
void Wolf::stealWeapon(Warrior* Enemy,int cityorder) {//狼抢走武器
	Enemy->OwnArsenal.Sort_When_Loss_Or_Steal();//被抢之前排序
	if (Enemy->OwnArsenal.WeaponNumber() == 0||OwnArsenal.WeaponNumber()==10) {//如果敌人没有武器,或者狼的武器已经满了
		return;
	}
	int minorder;
	if (Enemy->OwnArsenal.myWeapon[0]->order == 0) {
		minorder = 0;//敌人的武器库中的第 1 件武器的编号如果是0
	}
	else if (Enemy->OwnArsenal.myWeapon[0]->order == 1) {
		minorder = 1;//敌人的武器库中的第 1 件武器的编号如果是1
	}
	else {
		minorder = 2;//敌人的武器库中的第 1 件武器的编号如果是2
	}
	int myNum = OwnArsenal.WeaponNumber();//得到狼现有的武器数量
	int i = 0;
	for (; Enemy->OwnArsenal.myWeapon[i] != NULL && i + myNum < 10 && Enemy->OwnArsenal.myWeapon[i]->order == minorder; i++) {
		OwnArsenal.AddNewWeapon(Enemy->OwnArsenal.myWeapon[i]);
		Enemy->OwnArsenal.myWeapon[i] = NULL;
	}
	Enemy->OwnArsenal.PushFront();
	control::Timeprint();
	cout << (color == 0 ? "red " : "blue ") << ItsName << " " << order << " took " << i;
	switch (minorder) {
	case 0:cout << " sword"; break;
	case 1:cout << " bomb"; break;
	case 2:cout << " arrow";
	}
	cout << " from " << (Enemy->color == 0 ? "red " : "blue ") << Enemy->ItsName << " " << Enemy->order << " in city " << cityorder << endl;
	OwnArsenal.SortByOrder();//给狼自身排序
}
void City::CityWolfSteal() {//只有两种情况下狼会抢敌人武器
	//针对每一个城市，有这样的一个函数
	if (WarriorInIt[0]->ItsName == "wolf" && WarriorInIt[1]->ItsName != "wolf") {
		WarriorInIt[0]->stealWeapon(WarriorInIt[1],order);
	}
	if (WarriorInIt[1]->ItsName == "wolf" && WarriorInIt[0]->ItsName != "wolf") {
		WarriorInIt[1]->stealWeapon(WarriorInIt[0],order);
	}
}
void WolfsRobWeapons() {
	City* check_one_wolf = FirstCity;
	for (int i = 1; i < control::cityNum; i++) {
		if(check_one_wolf->WarriorInIt[0]!=NULL && check_one_wolf->WarriorInIt[1]!=NULL)
			check_one_wolf->CityWolfSteal();
		check_one_wolf = check_one_wolf->LatterCity;
	}
	if (check_one_wolf->WarriorInIt[0] != NULL && check_one_wolf->WarriorInIt[1] != NULL)
		check_one_wolf->CityWolfSteal();
}
//########################################################
	//排序时cmp将使用过的排在前面	//获取武器，
void get_Weapon_After_Win(Warrior* Winner, Warrior* Losser) {
	Losser->OwnArsenal.Sort_When_Loss_Or_Steal();
	int WinnerNum = Winner->OwnArsenal.WeaponNumber();
	int LosserNum = Losser->OwnArsenal.WeaponNumber();
	int newget = 0;
	while (WinnerNum == 10 || newget < LosserNum) {
		if (Losser->OwnArsenal.myWeapon[newget]!=NULL){
			Winner->OwnArsenal.AddNewWeapon(Losser->OwnArsenal.myWeapon[newget]);
		}//加入后自动使用SortByOrder排序
		else {
			return;
		}
		newget++;
		WinnerNum++;
	}
}
int City::FindAttacker() {//如果return 0 红方先攻击，如果 1 蓝方先攻击，如果 -1 无战斗；
	if (WarriorInIt[0] == NULL || WarriorInIt[1] == NULL) {
		return noflag;//-1
	}
	if (order % 2 == 1)//红旗或者为奇数
		return red;
	else
		return blue;
}
int City::CityJudgeResult() {// 0 红方胜利，1 蓝方胜利，-1 都死; -2都活 
	if (WarriorInIt[0]->blood <= 0 && WarriorInIt[1]->blood <= 0) {
		return -1;
	}
	if (WarriorInIt[0]->blood > 0 && WarriorInIt[1]->blood <= 0) {
		get_Weapon_After_Win(WarriorInIt[0], WarriorInIt[1]);//胜利的一方缴获武器
		return 0;
	}
	if (WarriorInIt[0]->blood <= 0 && WarriorInIt[1]->blood > 0) {
		get_Weapon_After_Win(WarriorInIt[1], WarriorInIt[0]);
		return 1;
	}
	else {//和平
		return -2;
	}
}
void City::CityDragonYell() {
	for (int i = 0; i < 2; i++) {
		if (WarriorInIt[i] != NULL && WarriorInIt[i]->ItsName=="dragon") {
			WarriorInIt[i]->TakeActions1();
			cout << order << endl;
		}
	}
}
int City::CityBeginWar(Warrior* Attacker,Warrior *Defender){
	Attacker->OwnArsenal.SortByOrder();
	Defender->OwnArsenal.SortByOrder();
	//if (debUg == 1&&WarriorInIt[0]->ItsName=="iceman"&&WarriorInIt[1]->ItsName=="dragon") {
	//	int dd = 0;
	//}
	int At_num = Attacker->OwnArsenal.WeaponNumber();
	int De_num = Defender->OwnArsenal.WeaponNumber();
	if (At_num == 0 && De_num == 0) {
		return -2;//双方无武器，战争以和平方式结束
	}
	//武器用完还活着；武器没有用完但是状态不改变
	int turns = 0;
	int curAtBlood = Attacker->blood;
	int curDeBlood = Defender->blood;
	while (turns ==0 || curAtBlood!=Attacker->blood||curDeBlood!= Defender->blood||
	(Attacker->OwnArsenal.myWeapon[0]!=NULL&& Attacker->OwnArsenal.myWeapon[0]->order!=0)
		||(Defender->OwnArsenal.myWeapon[0] != NULL && Defender->OwnArsenal.myWeapon[0]->order != 0)) {//血量不变（武器状态不变？）
		turns++;
		int At_Use_Order = 0; int De_Use_Order = 0;
		curAtBlood = Attacker->blood;
		curDeBlood = Defender->blood;
		int count = 0;
		while (Attacker->blood > 0 && Defender->blood > 0)//分别使用一轮武器库
		{//有一个问题：一方的武器只有一个，另一方有好多个，前一方不能一直挨打
			count++;
			if (At_Use_Order < At_num) {//使用一轮武器
				Attacker->OwnArsenal.myWeapon[At_Use_Order]->UsePower(Attacker, Defender);
				Attacker->OwnArsenal.DeleteUsedOne();
				At_Use_Order++;
			}
			if (Attacker->blood <= 0 || Defender->blood <= 0 || (At_Use_Order >= At_num && De_Use_Order >= De_num)) {
				//进行第二轮遍历双方的武器库
				break;
			}
			if (De_Use_Order < De_num) {
				Defender->OwnArsenal.myWeapon[De_Use_Order]->UsePower(Defender, Attacker);
				Defender->OwnArsenal.DeleteUsedOne();
				De_Use_Order++;
			}
			if (Attacker->blood <= 0 || Defender->blood <= 0 || (At_Use_Order >= At_num && De_Use_Order >= De_num)) {
				//进行第二轮遍历双方的武器库
				break;
			}
			if (At_Use_Order >= At_num && De_Use_Order < De_num) {
				Attacker->OwnArsenal.PushFront();
				At_num = Attacker->OwnArsenal.WeaponNumber();//更新攻击者的武器库
				At_Use_Order = 0;
			}
			else if (De_Use_Order >= De_num && At_Use_Order < At_num) {
				Defender->OwnArsenal.PushFront();
				De_num = Defender->OwnArsenal.WeaponNumber();//更新被攻击者的武器库
				De_Use_Order = 0;
			}
			//if (count > 100000) {
			//	break;
			//}
		}
		Attacker->OwnArsenal.PushFront();
		Defender->OwnArsenal.PushFront();//向前挪动
		if (Attacker->blood <= 0 || Defender->blood <= 0) {
			break;//有人死去，跳出循环
		}
		At_num = Attacker->OwnArsenal.WeaponNumber();//得到武器的数量
		De_num = Defender->OwnArsenal.WeaponNumber();
		if (At_num == 0 && De_num==0) {//武器全部用完
			break;
		}
	}
	//arrow 优先缴获没用过的，使用时优先使用用过的
	int result = CityJudgeResult();
	return result;
}
void City::printResult(int result) {
	control::Timeprint();
	if (result >= 0) {
		Warrior* Winner;
		Warrior* losser;
		if (result == 0) {
			Winner = WarriorInIt[0];
			losser = WarriorInIt[1];
		}
		else {
			Winner= WarriorInIt[1];
			losser = WarriorInIt[0];
		}
		cout << (Winner->color == 0 ? "red " : "blue ") << Winner->ItsName << " " << Winner->order << " killed" << (losser->color == 0 ? " red " : " blue ")
			<< losser->ItsName << " " << losser->order << " in city " << order << " remaining " << Winner->blood << " elements"<<endl;
	}
	else {
		cout << "both" << (WarriorInIt[0]->color == 0 ? " red " : " blue ") << WarriorInIt[0]->ItsName << " " << WarriorInIt[0]->order << " and "
			<< (WarriorInIt[1]->color == 0 ? "red " : "blue ") << WarriorInIt[1]->ItsName << " " << WarriorInIt[1]->order << (result == -1 ? " died" : " were alive")<< " in city "<<order << endl;
	}
	//埋葬死去的武士
	if (result == 0) {
		WarriorInIt[1] = NULL;
	}
	else if (result == 1) {
		WarriorInIt[0] = NULL;
	}
	else if (result == -1) {
		WarriorInIt[0] = NULL; WarriorInIt[1] = NULL;
	}
}
void City::War_and_Result() {//在某一座城市中发起战争并且输出结果
	int Attacker = FindAttacker();//Find Attacker and Defender();
	if (Attacker == noflag) {
		return;//no war;
	}
	//狼偷武器在之前已经完成过了
	int result;
	if (debUg == 2 && WarriorInIt[0]->ItsName == "dragon" && WarriorInIt[1]->ItsName == "ninja") {
		int de = 54;
	}
	if (Attacker == 0) {
		result = CityBeginWar(WarriorInIt[0], WarriorInIt[1]);
	}
	else {
		result = CityBeginWar(WarriorInIt[1], WarriorInIt[0]);
	}
	printResult(result);
	CityDragonYell();//这里是一个接口
}
void ReportWarCondition() {//连带 cheer up 一起执行
	City* ReportCity = FirstCity;
	for (int i = 1; i < control::cityNum; i++) {
		if(ReportCity->WarriorInIt[0]!=NULL && ReportCity->WarriorInIt[1]!=NULL)
			ReportCity->War_and_Result();
		ReportCity = ReportCity->LatterCity;
	}
	if(ReportCity->WarriorInIt[0]!=NULL && ReportCity->WarriorInIt[1]!=NULL)
		ReportCity->War_and_Result();
}
//##########################################################
void printElements(command * cm) {
	control::Timeprint();
	//000:50 100 elements in red headquarter
	cout << cm->TheForce << " elements in " << (cm->color == 0 ? "red" : "blue") << " headquarter" << endl;

}
void ReportTheForceOfCommand(command* red, command* blue) {
	printElements(red);
	printElements(blue);
}
//###############################################################
void ReportWeapon(Warrior* W) {
	int num = W->OwnArsenal.WeaponNumber();
	int swordNum = 0, bombNum = 0, arrowNum = 0;
	for (int i = 0; i < num; i++) {
		switch (W->OwnArsenal.myWeapon[i]->order) {
		case 0:swordNum++; break;
		case 1:bombNum++; break;
		case 2:arrowNum++;
		}
	}
	cout << swordNum << " sword " << bombNum << " bomb " << arrowNum << " arrow";
}
void ReportWarroirsSitutation(Warrior* W) {//参数为最靠近红方的城市，开始遍历，调用
	//000:55 blue lion 1 has 0 sword 1 bomb 0 arrow and 10 elements
	control::Timeprint();
	cout << (W->color == 0 ? "red " : "blue ") << W->ItsName << " " << W->order << " has ";
	ReportWeapon(W);
	cout << " and " << W->blood << " elements" << endl;;
}
void ReportWarroirsByCity() {
	City* ReportCity = FirstCity;
	for (int i = 1; i < control::cityNum; i++) {
		for (int k = 0; k < 2; k++) {
			if (ReportCity->WarriorInIt[k] != NULL) {
				//cout << ReportCity->WarriorInIt[k]->order << endl;
				ReportWarroirsSitutation(ReportCity->WarriorInIt[k]);
			}
		}
		ReportCity = ReportCity->LatterCity;
	}
	for (int k = 0; k < 2; k++) {
		if (ReportCity->WarriorInIt[k] != NULL) {
			ReportWarroirsSitutation(ReportCity->WarriorInIt[k]);
		}
	}
}
//##############################################################
void HappenInAnHour(command *red ,command *blue,City *const FirstCity,int casenum) {//游戏的主流程
	while (control::time <= TotalTime) {
		WarriorsBorn(red,blue);
		control::timeAdd(5);
		if (control::time > TotalTime) {
			break;
		}
		LionRunAway();
		control::timeAdd(5);//每小时的第十分钟
		if (control::time > TotalTime) {
			break;
		}
		int If_gameover=WarriorsMoveForward();//武士前进，并检查是否有武士到达敌方司令部
		if (If_gameover != 0) {
			return;
		}
		control::timeAdd(25);
		if (control::time > TotalTime) {
			break;
		}
		WolfsRobWeapons();
		control::timeAdd(5);//每小时的第40分钟
		if (control::time > TotalTime) {
			break;
		}
		ReportWarCondition();//发生战斗40分钟
		control::timeAdd(10);//
		if (control::time > TotalTime) {
			break;
		}
		ReportTheForceOfCommand(red,blue); //每小时的第50分钟
		control::timeAdd(5);
		if (control::time > TotalTime) {
			break;
		}
		ReportWarroirsByCity();//55分钟
		control::timeAdd(5);
	}
	return;
}
//剩余：战斗和报告战斗顺序，武器库数据结构的完善，明确占领司令部的顺序，明确狼偷武器的正确性
//初始化静态变量；
int control::time = 0;
int control::cityNum = 0;

int main() {
	int caseNumber;
	cin >> caseNumber;//对应输入t
	for (int ca = 1; ca <= caseNumber; ca++) {
		cout << "Case " << ca <<':'<< endl;
		int originForce;
		cin >> originForce;//对应M
		command* red=new command(originForce,0);
		command* blue=new command(originForce,1);//创建两个司令部
		//要求：先输出红司令部的，然后输出蓝司令部的；
		cin >> control::cityNum;//对应N
		FirstCity = CitiesBuild(control::cityNum);
		cin >> LoyalityDecrease;//对应K
		cin >> TotalTime;//对应T
		control::getBloodArray();//得到各类武士的初始血量
		control::getStrengthArray();
		HappenInAnHour(red, blue,FirstCity,ca);
		NewCase();
		redHenemy = NULL;
		blueHenemy = NULL;
		control::time = 0;
	}
	return 0;
}