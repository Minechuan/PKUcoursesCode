/*
统一：在打完仗之后统计武器，并清楚已经使用过的武器
//使用炸弹与忍者的关系？
//即使去除无效的武器
清除武器
*/
#include<iostream>
#include<string>
#include<iomanip>
#include<algorithm>
#include<fstream>
using namespace std;
class Arsenal;
class Warrior;
class Sword;
class Bomb;
class Arrow;
class City;
class control;
class command;
Warrior* redHenemy[2];//占领两次
Warrior* blueHenemy[2];//敌军士兵会占领这个司令部

/*********************************************
******			   全局变量				******
*********************************************/
int BloodArray[5] = { 0,0,0,0,0 };
int StrengthArray[5] = { 0,0,0,0,0 };
string WeaponArray[3] = { "sword","bomb","arrow" };
Warrior* redHeadquarterW = NULL;
Warrior* blueHeadquarterW = NULL;
int LoyalityDecrease = 0;//每轮狮子如果被打败减少的忠诚值
int TotalTime = 0;
City* FirstCity = NULL;
City* LastCity = NULL;
int ArrowForce;

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
int control::time = 0;
int control::cityNum = 0;
void control::getBloodArray() {
	int temp = 0;
	for (int i = 0; i < 5; i++) {
		std::cin >> temp;
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
	cout << hours << ":" << setw(2) << setfill('0') << time % 60 << ' ';
}
/************************
*		武器				*
*		武器库			*
************************/
class WeaPPP {

};
int Debug1 = 0;
class Sword :public WeaPPP {
private:
	int Force;
public:
	Sword(int WAStrength) { Force = WAStrength * 2 / 10; }
	int UsePower(bool real = true) {
		if (real == false) {
			return Force;
		}
		int tmp = Force;
		Force = int(Force * 8 / 10);//需要注意，何时判断了
		return tmp;
	};//传入两个指向武士的指针,使用方法在武士类的定义之后
	int getForce() {
		return Force;
	}
};
class Bomb :public WeaPPP {
public:
	Bomb() {}
	bool UsePower(Warrior* Attacker, Warrior* Defender, bool myPos);
};
class Arrow :public WeaPPP {
public:
	int UseTimes;
	Arrow() :UseTimes(3) {}
	bool UsePower(Warrior* Defender);
	int getUseTime() {
		return UseTimes;
	}
};
//#####################################################################
class Arsenal {//其实也可以使用STL中的优先队列
public:
	//可以用三个长度为十的指针数组来存放现有的武器；
	Sword* mySword;
	Bomb* myBomb;
	Arrow* myArrow;
	void DeleteUsedOne();
	Arsenal() {
		mySword = nullptr;
		myBomb = nullptr;
		myArrow = nullptr;
	}//初始化
};
void Arsenal::DeleteUsedOne() {//删除一个已经使用完的武器,设置为空指针
	if (mySword != nullptr && mySword->getForce() == 0) {
		mySword = nullptr;
	}
	if (myArrow != nullptr && myArrow->getUseTime() == 0) {
		myArrow = nullptr;
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
	void ReportWeapon();//报告武器情况
	void ReportBlood();
	int  Attack(bool real = true);
	void Hurt(const int& lossblood);
	virtual int FightBack(bool real = true);//虚函数，有差异性
	virtual void GainWeaponWhenBorn(int order, int strength) {};//虚函数
	virtual void TakeActions1() {}//狮子减少Loyality
	virtual bool TakeActions2() { return 0; };//武士欢呼或者狮子逃跑//能够调用到派生类中个性化的成员函数；需要在派生类中也定义虚函数连接；
	virtual void stealWeapon(Warrior* Enemy) {};//实际上只针对wolf
	virtual void Change_Morale(int result) {};//只针对Dragon
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
void Warrior::ReportWeapon() {//报告武士的情况
	control::Timeprint();
	cout << (this->color == 0 ? "red " : "blue ") << ItsName << ' ' << order << " has ";
	if (OwnArsenal.myArrow != nullptr) {
		if (OwnArsenal.myBomb != nullptr) {
			cout << "arrow(" << OwnArsenal.myArrow->getUseTime() << "),bomb";
			if (OwnArsenal.mySword != nullptr) {
				cout << ",sword(" << OwnArsenal.mySword->getForce() << ")";
			}
		}
		else {
			cout << "arrow(" << OwnArsenal.myArrow->getUseTime() << ")";
			if (OwnArsenal.mySword != nullptr) {
				cout << ",sword(" << OwnArsenal.mySword->getForce() << ")";
			}
		}
	}
	else {
		if (OwnArsenal.myBomb != nullptr) {
			cout << "bomb";
			if (OwnArsenal.mySword != nullptr) {
				cout << ",sword(" << OwnArsenal.mySword->getForce() << ")";
			}
		}
		else {
			if (OwnArsenal.mySword != nullptr) {
				cout << "sword(" << OwnArsenal.mySword->getForce() << ")";
			}
			else {
				cout << "no weapon";
			}
		}
	}
	cout << endl;
}
void Warrior::ReportBlood() {
	cout << "with " << blood << " elements and force " << strength << endl;
}
//刚出生得到武器,并且将它们加入到武器库中
int Warrior::Attack(bool real) {//只针对sword
	if (blood <= 0) {
		return 0;
	}
	if (OwnArsenal.mySword != nullptr) {
		return OwnArsenal.mySword->UsePower(real) + strength;
	}
	return strength;
}
void Warrior::Hurt(const int& lossblood) {
	this->blood -= lossblood;
}
int Warrior::FightBack(bool real) {
	if (blood <= 0) {
		return 0;//已死，无伤害
	}
	if (OwnArsenal.mySword != nullptr) {
		return OwnArsenal.mySword->UsePower(real) + strength / 2;
	}
	return strength / 2;
}
//dragon 、ninja、iceman、lion、wolf 为生成数组中对应的顺序；
// 0        1      2      3      4
// 
//需要注意，在函数中构造的变量不能在函数消亡时析构！
//那么现有的变量应该如何维护，既然武士有一个编号，那么可以根据编号存放在一个可变数组中；即以Warrior为元素的数组
//确定到底是哪一方进行建造，使用一个 bool 类型 color，如果为 0，那么红方建造，反之 1 蓝方建造； 
bool Bomb::UsePower(Warrior* Attacker, Warrior* Defender, bool myPos) {//与具体的战斗是分离的两人一人判断一次
	//myPos为1，说明他是之后战斗的攻击者
	//若为0为反击者
	//若返回为True，结论为使用炸弹
	if (myPos == 1) {//决策者是先发攻击者
		if (Attacker->blood <= 0) {
			return false;
		}//先判断是否会先发制人
		if (Defender->blood <= Attacker->Attack(false)) {
			return false;
		}
		if (Attacker->blood <= Defender->FightBack(false)) {
			return true;
		}
	}
	else {
		if (Defender->blood <= 0) {
			return false;
		}
		if (Defender->blood <= Attacker->Attack(false)) {
			return true;
		}
	}
	return false;
}
bool Arrow::UsePower(Warrior* Defender) {
	Defender->blood -= ArrowForce;
	UseTimes--;
	if (Defender->blood <= 0) {
		return true;
	}
	else {
		return false;
	}
}
//		 ############
//########	Lion	#######//
//		 ############
class Lion :public Warrior {
private:
	static const int gen;
public:
	int Loyalty = 0;
	static const string name;
	Lion(int blood, int strength, int color, int order, int TheForce) : Warrior(blood, strength, color, order, this->name, this->gen), Loyalty(TheForce) {
		cout << "Its loyalty is " << Loyalty << endl;
	}
	virtual void TakeActions1();//降低忠实值
	virtual bool TakeActions2();//决定是否欢呼
};
const int Lion::gen = 3;
const string Lion::name = "lion";
void Lion::TakeActions1() {//用来减少忠诚值
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
	Wolf(int blood, int strength, int color, int order) : Warrior(blood, strength, color, order, this->name, this->gen) {}
	virtual void stealWeapon(Warrior* Enemy);
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
		cout << "Its morale is " << fixed << setprecision(2) << Morale << endl;
		GainWeaponWhenBorn(order, strength);
	}
	virtual void GainWeaponWhenBorn(int order, int strength);
	virtual bool TakeActions2();
	virtual void Change_Morale(int result);
};
const int Dragon::gen = 0;
const string Dragon::name = "dragon";
void Dragon::GainWeaponWhenBorn(int order, int strength) {
	int weapon_order = order % 3;
	switch (weapon_order)
	{
	case 0:OwnArsenal.mySword = new Sword(strength); break;
	case 1:OwnArsenal.myBomb = new Bomb; break;
	case 2:OwnArsenal.myArrow = new Arrow;
	default:
		break;
	};
}
bool Dragon::TakeActions2() {
	if (Morale > 0.8) {//龙欢呼之前最后的判断
		control::Timeprint();
		cout << (color == 0 ? "red " : "blue ") << ItsName << " " << order << " yelled in city ";
		return true;
	}
	return false;
}
void Dragon::Change_Morale(int result) {//result取得胜利的一方
	if (result == color) {//取得胜利
		Morale += 0.2;
	}
	else if (result == 1 - color) {//未能获胜
		Morale -= 0.2;
	}
}
//		 ############
//########  Iceman	#######//
//		 ############
class Iceman :public Warrior {
private:
	int One_or_two_Step;
	static const int gen;
	static const string name;
public:
	Iceman(int blood, int strength, int color, int order) :
		Warrior(blood, strength, color, order, Iceman::name, this->gen) {
		GainWeaponWhenBorn(order, strength);
	}
	virtual void GainWeaponWhenBorn(int order, int strength);
	void virtual TakeActions1();
};
const string Iceman::name = "iceman";
const int Iceman::gen = 2;//对应的编号
void Iceman::GainWeaponWhenBorn(int order, int strength) {
	int weapon_order = order % 3;
	switch (weapon_order)
	{
	case 0:OwnArsenal.mySword = new Sword(strength); break;
	case 1:OwnArsenal.myBomb = new Bomb; break;
	case 2:OwnArsenal.myArrow = new Arrow; break;
	default:
		break;
	};
}
void Iceman::TakeActions1() {
	if (One_or_two_Step == 0) {
		One_or_two_Step++;
	}
	else {
		if (blood > 9) {
			blood -= 9;
		}
		else {
			blood = 1;
		}
		strength += 20;
		One_or_two_Step = 0;
	}
}
//		 ############
//########	Ninja	#######//
//		 ############
class Ninja :public Warrior {
private:
	static const int gen;
	static const string name;
public:
	Ninja(int blood, int strength, int color, int order) : Warrior(blood, strength, color, order, this->name, this->gen) {
		GainWeaponWhenBorn(order);
		GainWeaponWhenBorn(order + 1);
	}
	virtual void GainWeaponWhenBorn(int order);
	int FightBack(bool real = true) { return 0; };//不造成反击伤害
};
const int Ninja::gen = 1;
const string Ninja::name = "ninja";
void Ninja::GainWeaponWhenBorn(int order) {
	int weapon_order = order % 3;
	switch (weapon_order)
	{
	case 0:OwnArsenal.mySword = new Sword(strength); break;
	case 1:OwnArsenal.myBomb = new Bomb; break;
	case 2:OwnArsenal.myArrow = new Arrow; break;
	default:
		break;
	};
}
//明确 “is a ” 和 “has a ”的关系，确定成员应该放在哪个类中；
/*******************************
***	    司令部Headquarter	****
*******************************/
class command {//一共只需要两个对象！
private:
	int BornArray[5];//产生顺序
public:
	int TheForce = 0;//目前的晶元数量
	int color;//表示为哪一方的司令部
	int TotalNumCreated = 0;//目前两个司令部累计创建的士兵数量，用来给每个阵营创造的武士编号
	int buildTurn = 0;//目前按照顺序生产到第几个
	command(int ori, int colorIn);
	bool canBuild();//判断是否能生产
	//int findWhich() ;//按照顺序查看，能创造哪个就返回那个对象！希望是可以的
	void Build();//确定能生产之后，决定生产那一个
	void add_element(Warrior* W, int num, bool Is_a_Warrior) {
		//000:30 blue lion 1 earned 10 elements for his headquarter
		if (Is_a_Warrior == true) {
			control::Timeprint();
			cout << (W->color == 0 ? "red " : "blue ") << W->ItsName << ' ' << W->order << " earned " << num << " elements for his headquarter" << endl;;
		}
		TheForce += num;
	}
	void addt(int num) {
		TheForce += num;
	}
};
static command* RedH = new command(0, 0);
static command* BlueH = new command(0, 1);//创建两个司令部
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
	if (TheForce < BloodArray[BornArray[buildTurn % 5]]) {//theforce 是command的对象；
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
	int gen = BornArray[buildTurn % 5];//确定能建造该武士；
	buildTurn++;
	TotalNumCreated++;
	switch (gen) {//分编号讨论
	case 0: {TheForce -= BloodArray[0]; Dragon* NewDragon = new Dragon(BloodArray[0], StrengthArray[0], color, TotalNumCreated, TheForce);
		(color == 0 ? redHeadquarterW = NewDragon : blueHeadquarterW = NewDragon);
		break; }
	case 1: {TheForce -= BloodArray[1]; Ninja* NewNinja = new Ninja(BloodArray[1], StrengthArray[1], color, TotalNumCreated);
		(color == 0 ? redHeadquarterW = NewNinja : blueHeadquarterW = NewNinja);
		break; }
	case 2: {TheForce -= BloodArray[2]; Iceman* NewIceman = new Iceman(BloodArray[2], StrengthArray[2], color, TotalNumCreated);
		(color == 0 ? redHeadquarterW = NewIceman : blueHeadquarterW = NewIceman);
		break; }
	case 3: {TheForce -= BloodArray[3]; Lion* NewLion = new Lion(BloodArray[3], StrengthArray[3], color, TotalNumCreated, TheForce);
		(color == 0 ? redHeadquarterW = NewLion : blueHeadquarterW = NewLion);
		break; }
	case 4: {TheForce -= BloodArray[4]; Wolf* NewWolf = new Wolf(BloodArray[4], StrengthArray[4], color, TotalNumCreated);
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
	int Element;//城市产生的晶元数量
	int order;//城市的编号
	int LastWiner;
	int flagA;//red: 0; blue: 1; None: -1
	int cur_result;//用于在一场战斗中保持结果

	enum flag { red = 0, blue = 1, noflag = -1, };
	//int TheForceNum;//城市中的晶元数量
	Warrior* WarriorInIt[2];//0~4代表武士种类，用于方便的在五个指针数组中查找，若为-1则为空，因为一个城市中只能有双方各一个武士；
	City* AheadCity;
	City* LatterCity;
	//其中WarriorInIt[0]为红方的，WarriorInIt[1]为蓝方的
	City(int order_);
	City& operator = (const City  city);
	int UseBomb();
	void CityLionRun();
	void CityDragonYell(int firstA);
	void flag_judge_and_change();
	void War_and_Result();
	int FindAttacker();
	void CityBeginWar(Warrior* Attacker, Warrior* Defender);//0 red win; 1 blue win; -1 平局
	void CityJudgeResult();
	void RewardW();
	void AddE();
	int flagTime;
};
City::City(int order_) {
	order = order_;
	AheadCity = NULL;
	LatterCity = NULL;
	WarriorInIt[0] = NULL;
	WarriorInIt[1] = NULL;
	Element = 0;
	//##################################
	LastWiner = -1;//无人胜利
	cur_result = -100;//默认为100；
	flagTime = 0;
	flagA = -1;
}
City& City::operator=(const City city) {
	order = city.order;
	flagA = city.flagA;
	AheadCity = city.AheadCity;
	LatterCity = city.LatterCity;
	LastWiner = city.LastWiner;
	flagTime = city.flagTime;
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
void WarriorsBorn_0(command* red, command* blue) {//武士在
	if (red->canBuild()) {
		red->Build();
	}
	if (blue->canBuild()) {
		blue->Build();
	}
	return;
}
//######################################################
void printRun(Warrior* W) {
	//001:05 blue lion 1 ran away
	control::Timeprint();
	cout << (W->color == 0 ? "red " : "blue ") << W->ItsName << " " << W->order << " ran away" << endl;
}
void City::CityLionRun() {
	for (int i = 0; i < 2; i++) {
		if (WarriorInIt[i] != NULL && WarriorInIt[i]->ItsName == "lion") {
			//cout << "LLLiiiooonnn" << endl;
			if (WarriorInIt[i]->TakeActions2()) {
				printRun(WarriorInIt[i]);
				WarriorInIt[i] = NULL;
			}
		}
	}
}
void LionRunAway_5() {
	//但是已经到达敌人司令部的lion不会逃跑。lion在己方司令部可能逃跑。
	if (redHeadquarterW != NULL && redHeadquarterW->ItsName == "lion" && redHeadquarterW->color == 0) {
		if (redHeadquarterW->TakeActions2()) {//降生之后就想逃跑
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
	if (blueHeadquarterW != NULL && blueHeadquarterW->ItsName == "lion" && blueHeadquarterW->color == 1) {
		if (blueHeadquarterW->TakeActions2()) {
			printRun(blueHeadquarterW);
			blueHeadquarterW = NULL;
		}
	}
}
//######################################################
int cur_step_inRed = 0;//检查抵达了几个敌方武士
int cur_step_inBlue = 0;
void printReach(Warrior* winW) {
	//001:10 red iceman 1 reached blue headquarter with 20 elements and force 30
	control::Timeprint();
	cout << (winW->color == 0 ? "red " : "blue ") << winW->ItsName << " " << winW->order << " reached " << (winW->color == 0 ? "blue" : "red") <<
		" headquarter with " << winW->blood << " elements and force " << winW->strength << endl;
}
void printConquer(int color) {
	control::Timeprint();
	cout << (color == 0 ? "red" : "blue") << " headquarter was taken" << endl;
}
bool WarriorsReachEnemyCommand(int color) {//输出顺序可能有问题
	bool GameOver = 0;
	if (color == 0 && redHenemy[cur_step_inRed] != NULL && redHenemy[cur_step_inRed]->color == 1) {
		printReach(redHenemy[cur_step_inRed]);
		cur_step_inRed++;
		if (cur_step_inRed == 2) {
			printConquer(0);
			GameOver = 1;
		}
	}
	if (color == 1 && blueHenemy[cur_step_inBlue] != NULL && blueHenemy[cur_step_inBlue]->color == 0) {
		printReach(blueHenemy[cur_step_inBlue]);
		cur_step_inBlue++;
		if (cur_step_inBlue == 2) {
			printConquer(1);
			GameOver = 1;
		}
	}
	return GameOver;
}
void LetThemMove() {
	//首先移动蓝方
	City* cur_city = FirstCity;
	if (cur_city->WarriorInIt[1] != NULL) {
		if (cur_city->WarriorInIt[1]->ItsName == "iceman") {
			cur_city->WarriorInIt[1]->TakeActions1();
		}
		redHenemy[cur_step_inRed] = cur_city->WarriorInIt[1];
		cur_city->WarriorInIt[1] = NULL;//抵达敌方司令部的狮子不会逃跑
	}
	for (int i = 1; i < control::cityNum; i++) {
		cur_city = cur_city->LatterCity;
		if (cur_city->WarriorInIt[1] != NULL) {
			if (cur_city->WarriorInIt[1]->ItsName == "iceman") {
				cur_city->WarriorInIt[1]->TakeActions1();
			}
			cur_city->AheadCity->WarriorInIt[1] = cur_city->WarriorInIt[1];
			cur_city->WarriorInIt[1] = NULL;
		}
	}//已经到达最后一座城市
	if (blueHeadquarterW != NULL) {
		if (blueHeadquarterW->ItsName == "iceman") {
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
		blueHenemy[cur_step_inBlue] = cur_city->WarriorInIt[0];
		cur_city->WarriorInIt[0] = NULL;//红方最前的武士移动至蓝方司令部
	}
	for (int i = 1; i < control::cityNum; i++) {
		cur_city = cur_city->AheadCity;//自东向西遍历
		if (cur_city->WarriorInIt[0] != NULL) {
			if (cur_city->WarriorInIt[0]->ItsName == "iceman") {
				cur_city->WarriorInIt[0]->TakeActions1();
			}
			cur_city->LatterCity->WarriorInIt[0] = cur_city->WarriorInIt[0];
			if (cur_city->LatterCity->WarriorInIt[0]->ItsName == "iceman" && cur_city->LatterCity->order == 5) {

			}
			cur_city->WarriorInIt[0] = NULL;
		}
	}
	if (redHeadquarterW != NULL) {
		if (redHeadquarterW->ItsName == "iceman") {
			redHeadquarterW->TakeActions1();
		}
		cur_city->WarriorInIt[0] = redHeadquarterW;
		//cout << cur_city->WarriorInIt[0]->ItsName <<" "<<cur_city->WarriorInIt[1]->ItsName<< endl;
		redHeadquarterW = NULL;
	}
	//至此武士已经都已经前进到
}
void printMarchCity(Warrior* W, int Cityorder) {
	control::Timeprint();
	cout << (W->color == 0 ? "red " : "blue ") << W->ItsName << " " << W->order << " marched to city " << Cityorder << " with " << W->blood
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

	int If_Gameover2 = WarriorsReachEnemyCommand(1);//蓝方司令部是否被占领
	return If_Gameover1 | If_Gameover2;//只要有一方被占领
}
int WarriorsMoveForward_10() {
	LetThemMove();//使每个武士前进
	int ifg = report_Move_result();//前进之后按照顺序报告结果
	return ifg;//int 判断游戏是否结束
}
//########################################################
void ElementsProduct_20() {
	City* NoOne = FirstCity;
	for (int i = 1; i < control::cityNum; i++) {
		NoOne->Element += 10;
		NoOne = NoOne->LatterCity;
	}
	NoOne->Element += 10;
}
//####################################################

void getThem(Warrior* OnlyGuy, int num) {//获取有两种方式，这里是战斗开始之前城市里面一个武士和平传送的
	if (OnlyGuy->color == 0) {
		RedH->add_element(OnlyGuy, num, true);
	}
	else {
		BlueH->add_element(OnlyGuy, num, true);
	}
}
void judgeOnly(City* NoOne) {//只有两种可能
	if (NoOne->WarriorInIt[0] == NULL && NoOne->WarriorInIt[1] != NULL) {
		getThem(NoOne->WarriorInIt[1], NoOne->Element);
		NoOne->Element = 0;
	}
	else if (NoOne->WarriorInIt[0] != NULL && NoOne->WarriorInIt[1] == NULL) {
		getThem(NoOne->WarriorInIt[0], NoOne->Element);
		NoOne->Element = 0;
	}
}
void OneWarriorGetElements_30() {
	City* NoOne = FirstCity;
	for (int i = 1; i < control::cityNum; i++) {
		judgeOnly(NoOne);
		NoOne = NoOne->LatterCity;
	}
	judgeOnly(NoOne);
}
//##################################################
// 000:35 blue dragon 1 shot
void printShot(Warrior* S, Warrior* Death, bool if_kill) {
	control::Timeprint();//如果杀了人有不同的输出
	cout << (S->color == 0 ? "red " : "blue ") << S->ItsName << ' ' << S->order << " shot";
	if (if_kill == false) {
		cout << endl;
		return;
	}//and killed red lion 4
	cout << " and killed " << (Death->color == 0 ? "red " : "blue ") << Death->ItsName << ' ' << Death->order << endl;
}
void UseArrows_35() {//即使同时被射死，也不影响另一方放箭，所以这里就不清楚尸体了。
	City* cur = FirstCity;
	bool if_kill = false;
	if (cur->WarriorInIt[0] != NULL && cur->WarriorInIt[0]->OwnArsenal.myArrow != nullptr && cur->LatterCity->WarriorInIt[1] != NULL) {
		if_kill = cur->WarriorInIt[0]->OwnArsenal.myArrow->UsePower(cur->LatterCity->WarriorInIt[1]);
		printShot(cur->WarriorInIt[0], cur->LatterCity->WarriorInIt[1], if_kill);
		cur->WarriorInIt[0]->OwnArsenal.DeleteUsedOne();
	}
	cur = cur->LatterCity;//move pointer
	for (int i = 1; i < control::cityNum - 1; i++) {
		if (cur->WarriorInIt[0] != NULL && cur->WarriorInIt[0]->OwnArsenal.myArrow != nullptr && cur->LatterCity->WarriorInIt[1] != NULL) {
			if_kill = cur->WarriorInIt[0]->OwnArsenal.myArrow->UsePower(cur->LatterCity->WarriorInIt[1]);
			printShot(cur->WarriorInIt[0], cur->LatterCity->WarriorInIt[1], if_kill);
			cur->WarriorInIt[0]->OwnArsenal.DeleteUsedOne();
		}//当前城市有武士且武士有箭，射程上有敌人
		if (cur->WarriorInIt[1] != NULL && cur->WarriorInIt[1]->OwnArsenal.myArrow != nullptr && cur->AheadCity->WarriorInIt[0] != NULL) {
			if_kill = cur->WarriorInIt[1]->OwnArsenal.myArrow->UsePower(cur->AheadCity->WarriorInIt[0]);
			printShot(cur->WarriorInIt[1], cur->AheadCity->WarriorInIt[0], if_kill);
			cur->WarriorInIt[1]->OwnArsenal.DeleteUsedOne();
		}
		cur = cur->LatterCity;
	}
	if (cur->WarriorInIt[1] != NULL && cur->WarriorInIt[1]->OwnArsenal.myArrow != nullptr && cur->AheadCity->WarriorInIt[0] != NULL) {
		if_kill = cur->WarriorInIt[1]->OwnArsenal.myArrow->UsePower(cur->AheadCity->WarriorInIt[0]);
		printShot(cur->WarriorInIt[1], cur->AheadCity->WarriorInIt[0], if_kill);
		cur->WarriorInIt[1]->OwnArsenal.DeleteUsedOne();
	}

}
//##############################################
static void printExplosion(City* C, int res) {// you should delete the body
	if (res == -1) {
		return;
	}
	// 000:38 blue dragon 1 used a bomb and killed red lion 7
	control::Timeprint();
	cout << (C->WarriorInIt[res]->color == 0 ? "red " : "blue ") << C->WarriorInIt[res]->ItsName << ' ' << C->WarriorInIt[res]->order <<
		" used a bomb and killed " << (C->WarriorInIt[1 - res]->color == 0 ? "red " : "blue ") << C->WarriorInIt[1 - res]->ItsName << ' ' << C->WarriorInIt[1 - res]->order << endl;
	C->WarriorInIt[res] = NULL;//除去尸体不再考虑
	C->WarriorInIt[1 - res] = NULL;
	//C->LastWiner = -1;//不清除连续性，不算是一场战斗
}
int City::UseBomb() {
	if (WarriorInIt[0] != NULL && WarriorInIt[1] != NULL &&
		(WarriorInIt[0]->OwnArsenal.myBomb != nullptr || WarriorInIt[1]->OwnArsenal.myBomb != nullptr)) {//至少有一方有炸弹
		int ater = FindAttacker();//此时只能为0或1
		bool ifUse0 = false;
		bool ifUse1 = false;
		if (WarriorInIt[0]->OwnArsenal.myBomb != nullptr) {//红方有炸弹
			ifUse0 = WarriorInIt[0]->OwnArsenal.myBomb->UsePower(WarriorInIt[ater], WarriorInIt[1 - ater], 1 - ater);
		}
		if (WarriorInIt[1]->OwnArsenal.myBomb != nullptr) {//蓝方有炸弹
			ifUse1 = WarriorInIt[1]->OwnArsenal.myBomb->UsePower(WarriorInIt[ater], WarriorInIt[1 - ater], ater);
		}
		if (ifUse0 && !ifUse1) {//红方使用了炸弹
			return 0;
		}
		else if (!ifUse0 && ifUse1) {//但是最终肯定只有一方使用了炸弹
			return 1;
		}
	}
	return -1;
}
void UseBomb_38() {
	City* cur = FirstCity;
	for (int i = 1; i < control::cityNum; i++) {
		cur = cur->LatterCity;//规则限制，不使用炸弹则不会同归于尽
		int	res = cur->UseBomb();//有人被炸死返回使用炸弹的color,-1则无人死亡,
		printExplosion(cur, res);
	}
	int	res = cur->UseBomb();//有人被炸死返回使用炸弹的color,-1则无人死亡
	printExplosion(cur, res);
}
//##############################################
void Wolf::stealWeapon(Warrior* Enemy) {//狼抢走武器
	//偷武器之前需要将无用的武器销毁,已经实现了销毁
	if (this->OwnArsenal.mySword == nullptr && Enemy->OwnArsenal.mySword != nullptr) {
		this->OwnArsenal.mySword = Enemy->OwnArsenal.mySword;
	}
	if (this->OwnArsenal.myArrow == nullptr && Enemy->OwnArsenal.myArrow != nullptr) {
		this->OwnArsenal.myArrow = Enemy->OwnArsenal.myArrow;
	}
	if (this->OwnArsenal.myBomb == nullptr && Enemy->OwnArsenal.myBomb != nullptr) {
		this->OwnArsenal.myBomb = Enemy->OwnArsenal.myBomb;
	}
}
//########################################################
void City::flag_judge_and_change() {
	if (LastWiner == -1) {
		flagTime = 0;
	}
	if (cur_result == -100) {//单独的一个武士被射死
		return;//没有战斗
	}
	else if (cur_result == -1) {//平局//都被后方的箭射死
		return;
	}
	else if (cur_result == -2) {//有平局不再连续,洗谁也没有吧对方打死
		LastWiner = -1;
		flagTime = 0;//有平局不再连续,洗谁也没有吧对方打死
		return;
	}
	else if (cur_result >= 0) {//只有result为 0 或 1 才有胜负插旗一说
		if (LastWiner >= 0 && LastWiner == cur_result) {//之前有人获胜
			if (flagTime == 1 && flagA!=cur_result) {//之前刚连续赢得一次,有必要更换旗帜
				flagA = cur_result;//更换旗帜
				control::Timeprint();
				cout << (flagA == 0 ? "red " : "blue ") << "flag raised in city " << order << endl;
			}
			flagTime++;
		}
		else if (LastWiner >= 0 && cur_result == 1 - LastWiner) {//胜负转换
			LastWiner = cur_result;
			flagTime = 1;//赢得一次
		}
		else if (LastWiner == -1) {//第一个赢家
			LastWiner = cur_result;
			flagTime = 1;
		}
	}
}
//战斗过程：1.FindAttacker 2.CityBeginWar 3.CityJudgeResult 4.CityDragonYell 5.delete useless weapon 6.:stealWeapon 7.delete body 8.reward winner
int City::FindAttacker() {//如果return 0 红方先攻击，如果 1 蓝方先攻击
	if (flagA == -1) {//即使都死了，也能判断先手方
		if (order % 2 == 1)//奇数红方先攻击
			return red;
		else
			return blue;
	}
	return flagA;
}
void PrintAttack(Warrior* w1, Warrior* w2, City* c) {
	control::Timeprint();
	//003:40 red lion 2 attacked blue lion 1 in city 3 with 10 elements and force 33
	cout << (w1->color == 0 ? "red " : "blue ") << w1->ItsName << ' ' << w1->order << " attacked ";
	cout << (w2->color == 0 ? "red " : "blue ") << w2->ItsName << ' ' << w2->order;
	cout << " in city " << c->order << " with " << w1->blood << " elements and force " << w1->strength << endl;
}
void PrintFightback(Warrior* w1, Warrior* w2, City* c) {
	//004:3 red lion 2 fought back against blue ninja 3 in city 4 shot
	control::Timeprint();
	cout << (w1->color == 0 ? "red " : "blue ") << w1->ItsName << ' ' << w1->order << " fought back against " << (w2->color == 0 ? "red " : "blue ") << w2->ItsName << ' ' << w2->order
		<< " in city " << c->order << endl;
}
void PrintKill(Warrior* body, City* c) {
	control::Timeprint();//001:40 red lion 2 was killed in city 1
	cout << (body->color == 0 ? "red " : "blue ") << body->ItsName << ' ' << body->order << " was killed in city " << c->order << endl;
}
void City::CityJudgeResult() {// 0 红方胜利，1 蓝方胜利，-1 都死; -2都活 
	if (WarriorInIt[0]->blood <= 0 && WarriorInIt[1]->blood <= 0) {
		cur_result = -1;//平局,两个人都死了(同时被前方的对手射死，有可能发生)
	}
	else if (WarriorInIt[0]->blood > 0 && WarriorInIt[1]->blood <= 0) {
		cur_result = 0;//红方胜利
	}
	else if (WarriorInIt[0]->blood <= 0 && WarriorInIt[1]->blood > 0) {
		cur_result = 1;//蓝方胜利
	}
	else {//和平,都没死
		cur_result = -2;//平局
	}
}
void City::CityDragonYell(int firstA) {//欢呼 res为胜利者,同时delete使用过的武器
	for (int i = 0; i < 2; i++) {
		if (WarriorInIt[i] != NULL && WarriorInIt[i]->ItsName == "dragon" && WarriorInIt[i]->blood > 0) {//没死并且是Dragon,士气值大于0.8
			WarriorInIt[i]->Change_Morale(cur_result);//士气值的改变与谁先发动攻击无关
			if (firstA == i) {//主动进攻方
				bool IfWin = WarriorInIt[i]->TakeActions2();//武士欢呼
				if (IfWin) {
					cout << order << endl;
				}
			}
		}
		if (WarriorInIt[i] != NULL) {
			WarriorInIt[i]->OwnArsenal.DeleteUsedOne();//清楚用过的武器
		}
	}
}
void City::CityBeginWar(Warrior* Attacker, Warrior* Defender) {
	int red_blue_lion[2] = { 0,0 };// 判断狮子把生命元给对方
	if (Attacker->ItsName == "lion" && Attacker->blood > 0) {
		red_blue_lion[Attacker->color] = Attacker->blood;
	}
	if (Defender->ItsName == "lion" && Defender->blood > 0) {
		red_blue_lion[Defender->color] = Defender->blood;
	}
	//###################################################################################
	if (Attacker->blood > 0 && Defender->blood > 0) {//didn't died
		int h1 = Attacker->Attack();//之前被箭射死的这里不打印 kill 
		Defender->blood -= h1;
		PrintAttack(Attacker, Defender, this);
		if (Defender->blood <= 0) {
			PrintKill(Defender, this);
		}
	}
	if (Defender->blood > 0 && Attacker->blood > 0 && Defender->ItsName != "ninja") {//didn't died
		int h2 = Defender->FightBack();
		Attacker->blood -= h2;
		PrintFightback(Defender, Attacker, this);
		if (Attacker->blood <= 0) {
			PrintKill(Attacker, this);
		}
	}
	//战斗结束
	//###############################################
	//一定会CityJudgeResult
	CityJudgeResult();//判断胜负，改变 cur_result 变量
	//有效战斗 cur_result 一定不会依然等于 -100
	//###################################################
	if (cur_result >= 0) {//负方狮子的血量给了获胜方；
		WarriorInIt[cur_result]->blood += red_blue_lion[1 - cur_result];
	}
	red_blue_lion[0] = 0;
	red_blue_lion[1] = 0;
	//############################################################
}
void RewardWarrior(Warrior* W) {
	W->blood += 8;
}
void City::RewardW() {
	//需要重新遍历城市，因为要优先找到接近敌方司令部的，简单之处：这里不需要输出
	if (cur_result == 1) {//blue获胜
		if (BlueH->TheForce < 8) {
			return;
		}
		RewardWarrior(WarriorInIt[1]);
		BlueH->TheForce -= 8;
	}
	else if (cur_result == 0) {//red获胜
		if (RedH->TheForce < 8) {
			return;
		}
		RewardWarrior(WarriorInIt[0]);
		RedH->TheForce -= 8;
	}
}
int FinalE[2] = { 0,0 };
void City::AddE() {//如果有一方获胜，司令部获得城市中全部的生命元
	if (cur_result == 1) {//blue获胜
		BlueH->add_element(WarriorInIt[1], Element, true);//这里仅仅用于输出
		FinalE[1] += Element;
		BlueH->TheForce -= Element;//最后统一加上，在每个城市时减去
		Element = 0;
	}
	else if (cur_result == 0) {//red方获胜
		RedH->add_element(WarriorInIt[0], Element, true);
		FinalE[0] += Element;
		RedH->TheForce -= Element;
		Element = 0;
	}

}
void deleteBody(City* c) {
	if (c->WarriorInIt[0] != NULL && c->WarriorInIt[0]->blood <= 0) {
		c->WarriorInIt[0] = NULL;
	}
	if (c->WarriorInIt[1] != NULL && c->WarriorInIt[1]->blood <= 0) {
		c->WarriorInIt[1] = NULL;
	}
}
void City::War_and_Result() {//在某一座城市中发起战争并且输出结果，两人都在
	int Attacker = FindAttacker();//一定返回 0 或 1，总能根据城市确定先手方
	if (Attacker == 0) {//红方先攻击
		CityBeginWar(WarriorInIt[0], WarriorInIt[1]);
	}
	else {//蓝方先攻击
		CityBeginWar(WarriorInIt[1], WarriorInIt[0]);
	}//一定会 CityBeginWar
	//
	//此时已经得到了战争的结果，存在了 cur_result 中
	//同时战斗结果已经打印过了
	CityDragonYell(Attacker);//这里是一个接口,同时去除无效的武器!!!!!主动进攻才欢呼
	AddE();//输出武士为司令部赢得生命元
	flag_judge_and_change();//改变旗帜
	//####################################################################
	if (cur_result >= -2) {
		for (int i = 0; i < 2; i++) {//狮子未能杀死敌人
			if (WarriorInIt[i] != NULL && WarriorInIt[i]->ItsName == "lion" && cur_result != i) {//狮子没有杀死敌人
				WarriorInIt[i]->TakeActions1();
			}
		}
	}
	//此时考虑获胜的狼#######################################################
	if (cur_result >= 0) {//决出胜负，注意：！！！！！这里将尸体 delete 掉
		if (WarriorInIt[cur_result] != NULL && WarriorInIt[cur_result]->ItsName == "wolf" && WarriorInIt[cur_result]->blood > 0) {
			WarriorInIt[cur_result]->stealWeapon(WarriorInIt[1 - cur_result]);
		}
	}
}

void Fight_and_FightBack_40() {
	//发起战斗，得到结果，消除尸体，武士欢呼，狼抢夺武器
	City* ReportCity = FirstCity;
	for (int i = 1; i < control::cityNum; i++) {
		//if (control::time >= 1500 && ReportCity->order == 10) {
		ReportCity->cur_result = -100;//如果一人被射死但尸体还没有清除，也会进入 War_and_Result();
		if (ReportCity->WarriorInIt[0] != NULL && ReportCity->WarriorInIt[1] != NULL)//两人都在，发起战争，这里才有旗帜的转变
			ReportCity->War_and_Result();
		deleteBody(ReportCity);//再次清除尸体//如果有一个武士而且被射死，不影响旗帜
		ReportCity = ReportCity->LatterCity;
	}
	ReportCity->cur_result = -100;
	if (ReportCity->WarriorInIt[0] != NULL && ReportCity->WarriorInIt[1] != NULL)//两人都在，发起战争
		ReportCity->War_and_Result();//此时ReportCity已经到了最后一个城市
	deleteBody(ReportCity);//再次清除尸体
	//########################################################################################
	//从东向西遍历
	City* AllFor_red_reward = ReportCity;//奖励红方武士
	for (int i = 1; i < control::cityNum; i++) {
		if (AllFor_red_reward->cur_result == 0)
			AllFor_red_reward->RewardW();
		AllFor_red_reward = AllFor_red_reward->AheadCity;
	}
	if (AllFor_red_reward->cur_result == 0)
		AllFor_red_reward->RewardW();
	//从西向东遍历
	ReportCity = FirstCity;//奖励蓝方武士
	for (int i = 1; i < control::cityNum; i++) {
		if (ReportCity->cur_result == 1) {
			ReportCity->RewardW();
		}
		ReportCity->cur_result = -100;
		ReportCity = ReportCity->LatterCity;
	}
	if (ReportCity->cur_result == 1) {
		ReportCity->RewardW();
	}
	ReportCity->cur_result = -100;//清空本次战争结果的数据
	//##############################################################################################
	//这里才将武士赢得的生命元真正传送到司令部
	RedH->TheForce += FinalE[0];//清空累计的晶元
	FinalE[0] = 0;
	BlueH->TheForce += FinalE[1];
	FinalE[1] = 0;
}
//##########################################################
void printElements(command* cm) {
	control::Timeprint();
	//000:50 100 elements in red headquarter
	cout << cm->TheForce << " elements in " << (cm->color == 0 ? "red" : "blue") << " headquarter" << endl;

}
void ReportTheForceOfCommand_50(command* red, command* blue) {
	printElements(red);
	printElements(blue);
}
//###############################################################
void ReportWarroirsByCity_55() {
	City* ReportCity = FirstCity;
	for (int i = 1; i < control::cityNum; i++) {
		if (ReportCity->WarriorInIt[0] != NULL) {
			ReportCity->WarriorInIt[0]->OwnArsenal.DeleteUsedOne();
			ReportCity->WarriorInIt[0]->ReportWeapon();
		}
		ReportCity = ReportCity->LatterCity;
	}
	if (ReportCity->WarriorInIt[0] != NULL) {
		ReportCity->WarriorInIt[0]->OwnArsenal.DeleteUsedOne();
		ReportCity->WarriorInIt[0]->ReportWeapon();
	}

	if (blueHenemy[0] != NULL) {//蓝方司令部里面的红方士兵
		blueHenemy[0]->OwnArsenal.DeleteUsedOne();
		blueHenemy[0]->ReportWeapon();
	}
	//报告蓝武士
	if (redHenemy[0] != NULL) {
		redHenemy[0]->OwnArsenal.DeleteUsedOne();
		redHenemy[0]->ReportWeapon();
	}
	ReportCity = FirstCity;
	for (int i = 1; i < control::cityNum; i++) {
		if (ReportCity->WarriorInIt[1] != NULL) {
			ReportCity->WarriorInIt[1]->OwnArsenal.DeleteUsedOne();
			ReportCity->WarriorInIt[1]->ReportWeapon();
		}
		ReportCity = ReportCity->LatterCity;
	}
	if (ReportCity->WarriorInIt[1] != NULL) {
		ReportCity->WarriorInIt[1]->OwnArsenal.DeleteUsedOne();
		ReportCity->WarriorInIt[1]->ReportWeapon();
	}

}
//##############################################################
void HappenInAnHour(command* red, command* blue, City* const FirstCity, int casenum) {//游戏的主流程
	while (control::time <= TotalTime) {
		WarriorsBorn_0(red, blue);
		control::timeAdd(5);
		City* c = FirstCity;
		/*for (int i = 0; i < 5; i++) {
			c = c->LatterCity;
		}
		if (c->WarriorInIt[0] != NULL && c->WarriorInIt[0]->ItsName == "iceman" && c->WarriorInIt[0]->order == 9) {
			cout << "hhh" << endl;
		}*/
		if (control::time > TotalTime) { break; }
		LionRunAway_5();
		control::timeAdd(5);//每小时的第十分钟
		if (control::time > TotalTime) { break; }
		int If_gameover = WarriorsMoveForward_10();//武士前进，并检查是否有武士到达敌方司令部
		if (If_gameover != 0) {
			return;
		}
		control::timeAdd(10);
		if (control::time > TotalTime) { break; }
		ElementsProduct_20();
		control::timeAdd(10);
		if (control::time > TotalTime) { break; }
		OneWarriorGetElements_30();
		control::timeAdd(5);
		if (control::time > TotalTime) { break; }
		UseArrows_35();
		control::timeAdd(3);
		if (control::time > TotalTime) { break; }
		UseBomb_38();
		control::timeAdd(2);
		if (control::time > TotalTime) { break; }
		Fight_and_FightBack_40();
		control::timeAdd(10);
		if (control::time > TotalTime) { break; }
		ReportTheForceOfCommand_50(red, blue);
		control::timeAdd(5);
		if (control::time > TotalTime) { break; }
		ReportWarroirsByCity_55();//55分钟
		control::timeAdd(5);
	}
	return;
}
//剩余：战斗和报告战斗顺序，武器库数据结构的完善，明确占领司令部的顺序，明确狼偷武器的正确性
//初始化静态变量；
void reset() {
	redHenemy[0] = NULL;
	redHenemy[1] = NULL;
	blueHenemy[0] = NULL;
	blueHenemy[1] = NULL;
	cur_step_inRed = 0;//检查抵达了几个敌方武士
	cur_step_inBlue = 0;
	FinalE[0] = 0;
	FinalE[1] = 0;
	FirstCity = NULL;
	LastCity = NULL;
	RedH = new command(0, 0);
	BlueH = new command(0, 1);
}
int main() {
	//freopen("output.txt", "w", stdout);
	int caseNumber;
	cin >> caseNumber;//对应输入t
	for (int ca = 1; ca <= caseNumber; ca++) {
		cout << "Case " << ca << ':' << endl;
		int originForce;
		cin >> originForce;//对应M
		RedH->addt(originForce);
		BlueH->addt(originForce);
		//要求：先输出红司令部的，然后输出蓝司令部的；
		cin >> control::cityNum;//对应N
		FirstCity = CitiesBuild(control::cityNum);
		cin >> ArrowForce;
		cin >> LoyalityDecrease;//对应K
		cin >> TotalTime;//对应T
		control::getBloodArray();//得到各类武士的初始血量
		control::getStrengthArray();
		HappenInAnHour(RedH, BlueH, FirstCity, ca);
		reset();
		control::time = 0;
	}
	return 0;
}

//剩余任务完善战斗过程



//1015行
//shot的打印有问题