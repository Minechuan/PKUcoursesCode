#include<iostream>
#include<graphics.h>
#include<cstdlib>
#include<random>
#include<time.h>
#include<fstream>
using namespace std;
#define MAX_NUM 8//数量
#define MAX_SIZE 567//棋盘宽度
#define Thin_WIDE 7//横条宽度
#define small_SIZE 60//棋格边长
#define out_SIZE 12//边框宽度
//240,155,89
struct position {
	int x;
	int y;
};
struct ChessButton {
	int x;
	int y;
	int r;
	COLORREF hidecolor;//隐藏为和棋盘的颜色一样
	COLORREF choicecolor;//提示下棋者，人类可以落子的位置
	COLORREF incolor;//鼠标在里面的颜色
	COLORREF curcolor;//这个地方目前的
	int state;//1 代表激活，0 代表休眠，2代表已经有棋子之后无法落子。
};
struct EACHCHESS {
	int x;
	int y;
	int utility;//这个棋的效用
};
//一个重要的结构体，可以用来递归和回溯
struct TurnColor {//最终决定好将要翻转的点后，传出这个结构体
	position WillTurn[40];//判断需要翻转的点，
	int num;//一共有几个这样的点
};
TurnColor turncolor;


ChessButton allchess[MAX_NUM][MAX_NUM]{};
void drawButton(struct Button button);
void drawChessButton(struct ChessButton theButton);
void initializebutton();
int clickButton(struct Button* button, ExMessage msg);
int clickChessButton(struct ChessButton* chessbutton, ExMessage msg);
void HighlightTip(ChessButton* chess);
void WhoseTurn(Button* Infor, int information);
int mouseInButton(struct Button* button, ExMessage msg);
void FindPP(int color);
void UnHighlight(ChessButton* chess);
bool judge(int x, int y, int c);
void ButtonSleep();
int mouseInChessButton(struct ChessButton* button, ExMessage msg);
TurnColor AISTurn(int Colors, int x, int y, TurnColor& turncolor);
int CALcu();
void setAllchess();

using namespace std;
#define MAX_NUM 8//格子数
bool judge(int x, int y, int color); //判断是否可下,颜色为color者下棋
int ChessBoard[MAX_NUM][MAX_NUM]{};//黑棋为1，白棋为-1；没有下棋就为0；

const int directx[8] = { 1,0,-1, 0,1, 1,-1,-1 };//棋盘与方向
const int directy[8] = { 0,1, 0,-1,1,-1, 1,-1 };
int arr[MAX_NUM][MAX_NUM] = {
{800, -25,90,5,5,90,-25,800},
{-25,-25,80,1,1,80,-25,-25},
{90,  80,0,2,2,0,80,90},
{1,    1, 2,1,1,2,1,5},
{5,    1, 2,1,1,2,1,5},
{90,  80,0,2,2,0,80,90},
{-25,-25,80,1,1,80,-25,-25},
{800, -25,90,5,5,90,-25,800},
};
const int arrLater[MAX_NUM][MAX_NUM] = {
	{10000,-200,100, 5, 5,100,-200,10000},
	{-200,-100, 70,10,10,70,-100,-200},
	{100 ,  70,  0,10,10, 0,70,  100 },
	{5   ,  10, 10, 5, 5,10,10,5    },
	{  5   ,  10, 10, 5, 5,10,10,5 },
	{100 ,  70,  0,10,10, 0,70,  100  },
	{-200,-100, 70,10,10,70,-100,-200 },
	{10000,-200,100, 5, 5,100,-200,10000},
};
const int arrFinal[MAX_NUM][MAX_NUM] = {
	{500,-50,50,50,50,50,-50,500},
	{-50,-50,40,40,40,40 - 50,-50},
	  {40,30,40,30,30,40,30,40},
	  {5,1,2,1,1,2,1,5},
	  {5,1,2,1,1,2,1,5},
	{40,30,40,30,30,40,30,40},
	{-50,-50,40,40,40,40-50,-500},
	{500,-50,50,50,50,50,-50,50},
};
void changeArr() {//15步之后的权重数组
	for (int i = 0; i < MAX_NUM; i++) {
		for (int j = 0; j < MAX_NUM; j++) {
			arr[i][j] = arrLater[i][j];
		}
	}
	return;
}
void changeArrAgain() {//棋局接近尾声的权重数组
	for (int i = 0; i < MAX_NUM; i++) {
		for (int j = 0; j < MAX_NUM; j++) {
			arr[i][j] = arrFinal[i][j];
		}
	}
	return;
}
//权值可以使用权值表
//struct SOMEdate {//效用为以下两者的线性组合
// Part;//权值
// possibility;//下一步对方的行动力
//效用+对方最大的效用=排序标准，
//这样可能会形成套娃，为了递归能够return 计算四组后，把第五步的最大小用定为0,
//如果时间充足

int CALcu() {
	int Weight = 0;
	for (int k = 0; k < 8; k++) {
		for (int r = 0; r < 8; r++) {
			Weight += arr[k][r] * ChessBoard[k][r];
		}
	}
	return Weight;
}
int finEX = 0;
bool AIrev(int x, int y, int color, int Direction, TurnColor& AIturn) {//x,y为落子的坐标，color为被翻转棋子的颜色
	if (x < 0 || x >= 8 || y < 0 || y >= 8) { return 0; }//超出边界
	if (ChessBoard[x][y] == 0) { return 0; }//遇到空位
	if (ChessBoard[x][y] == -color) { return 1; }//被夹住，但是也可能有一种特殊情况两个棋连在一起，并没有夹子
	else {
		if (AIrev(x + directx[Direction], y + directy[Direction], color, Direction, AIturn)) {
			AIturn.WillTurn[AIturn.num].x = x;
			AIturn.WillTurn[AIturn.num].y = y;
			AIturn.num++;
			return 1;
		}
		return 0;
	}
}
TurnColor AISTurn(int Colors, int x, int y, TurnColor& AIturn) {//根据静态棋盘判断哪些棋将要被翻转
	for (int i = 0; i < 25; i++) {//color为需要被翻转的棋子颜色
		AIturn.WillTurn[i].x = 0;
		AIturn.WillTurn[i].y = 0;//初始化
	}
	AIturn.num = 0;//决定一共需要翻转几个棋子
	for (int i = 0; i < 8; i++) {//一共有八个方向需要判断
		int x1 = x + directx[i], y1 = y + directy[i];//某一个方向延伸
		AIrev(x1, y1, Colors, i, AIturn); //这个位置以color为颜色的棋将被翻转，i代表方向
	}
	return AIturn;
}

int MoveMent(int color) {//用来统计行动力，可以是0
	int MOVEnum = 0;
	//EACHCHESS ChessUtility[20]{ };//用来储存当前情况下的可行点
	for (int i = 0; i < MAX_NUM; i++) {
		for (int j = 0; j < MAX_NUM; j++) {
			if (ChessBoard[i][j] != 1 && ChessBoard[i][j] != -1 && allchess[i][j].state != 2) {//这些位置没有棋子
				if (judge(i, j, color)) {//这个位置color可以落子
					MOVEnum++;
				}
			}
		}
	}
	return MOVEnum;
}

bool judge(int x, int y, int color) {//判断是否可下,颜色为color者下棋
	int i;//八个方向
	for (i = 0; i < 8; i++) {//8个方向探索是否有子可吃
		bool gap = 0;//是否有间隔
		int x1 = x + directx[i], y1 = y + directy[i];// , tempchange = 0;
		while (1) {
			if (x1 < 0 || x1 >= 8 || y1 < 0 || y1 >= 8) { break; }//不能直接return，否则无法遍历8个方向
			if (ChessBoard[x1][y1] == 0)//还是空位
				break;
			else if (ChessBoard[x1][y1] == color) {//如果是自己的颜色
				if (gap) { return gap; }
				else { break; }
			}
			else {//是对方的棋//挨着对方的颜色
				gap = 1;
				x1 += directx[i];
				y1 += directy[i];
			}
		}
	}
	return 0;//pieceschange;
}
position ChoosePosition;
int Recursion = 0;//统计递归次数，有所区别
bool comPare(int *a,int *b) {//如果为true就剪去
	if (*a > 0 && *b > 0) {
		if (*a > 1.5 * *b) {
			return true;
		}
		else {
			if(*b>*a)*a = *b;
			return false;
		}
	}
	else if (*a < 0 && *b < 0) {
		if (*a > 0.6 * *b) {//需要剪
			return true;
		}
		else {
			if (*b > *a)*a = *b;
			return false;
		}
	}
	else if (*a >= 0 && *b <= 0) {
		return true;
	}
	else {//不能剪
		*a = *b;
		return false;
	}
}
int FindAI(int color) {//找到颜色为color可以落子的位置；开始为AI的颜色
	int finAI = 0; //判断是否有棋可下
	int num = 0;//第一个值存在1！
	EACHCHESS ChessUtility[40]{ };//当前步，最后需要进行排序
	for (int i = 0; i < MAX_NUM; i++) {//遍历棋盘，寻找可以落子的地方
		for (int j = 0; j < MAX_NUM; j++) {
			if (ChessBoard[i][j] != 1 && ChessBoard[i][j] != -1) {//这些位置没有棋子
				if (judge(i, j, color)) {//这个位置color可以落子
					num++;//可选择位置个数
					finAI++;//有棋可下
					int curWeight = CALcu();//当前权重和

					ChessBoard[i][j] = color;//模拟走下这步棋
					allchess[i][j].state = 2;
					Recursion++;//检查递归深度
					//模拟翻转，然后递归
					TurnColor AIturn;//定义结构体，用于储存将要翻转的棋
					AISTurn(-color, i, j, AIturn);//color为落子者的颜色，-color为被翻转者的颜色
					//得到被翻转的棋子坐标及个数
					for (int imire = 0; imire < AIturn.num; imire++) {//进行赋值
						ChessBoard[AIturn.WillTurn[imire].x][AIturn.WillTurn[imire].y] = color;//被颜色为color的一方翻转
					}
					//计算这一步之后棋盘的权重;
					int laterWeight = CALcu();
					if (Recursion == 4) {//算到k步之后不调用FindAI就不深入
						//cout << "five steps" << endl;//如果没棋可下，就不再深入,如果三步之内无棋可下
						ChessUtility[num].utility = 30*(laterWeight - curWeight) -  90* MoveMent(-color);
					}
					else { 
						
						int compareFirst = 30* (laterWeight - curWeight) - 80 * MoveMent(-color);//为了剪枝而提前判断局部效用函数
						if (i == 0 && j == 0 || i == 7 && j == 0 || i == 0 && j == 7 || i == 7 && j == 7) {
							compareFirst += 100000;
							cout << compareFirst << " ";
							//Sleep(100);
						}
						int MAXUUU = 0;//目前（前4个）的最大效用
						if (num < 3&&Recursion>=2&&compareFirst>MAXUUU&& Recursion >= 2) {//判断剪枝条件
							MAXUUU = compareFirst;//
						}
						if (num >= 3 && Recursion >= 2 && comPare(&MAXUUU,&compareFirst)){//MAXUUU为暂时的最大值可能会更新
							num--;//the position and it's later result will not be take into account;
							goto StopThinking;//递归深度为三之后开始剪枝，如果小太多，则不再深入
						}
						ChessUtility[num].x = i;
						ChessUtility[num].y = j;
						ChessUtility[num].utility =compareFirst-10*FindAI(-color);
						          //权重差———————————对方行动力——————————对方最优步的效用
					}
					cout << ChessUtility[num].utility << endl;
					//Sleep(100);
					//将假设翻转的棋翻转回来
StopThinking:		for (int imire = 0; imire < AIturn.num; imire++) {//进行赋值
						ChessBoard[AIturn.WillTurn[imire].x][AIturn.WillTurn[imire].y] = -color;
					}//被颜色为color的一方翻回来
					Recursion--;//深度撤回1
					ChessBoard[i][j] = 0;//回溯，原来没有棋
					allchess[i][j].state = 0;
				}
			}
		}
	}
	if (!finAI) {//FinAI如果是全局变量就不好处理
		//cout << "fehoijfwi" << endl;
		ChoosePosition.x = 9;
		ChoosePosition.y = 9;
		return 0;//一方无棋可下。可以直接减少搜索深度
	}
	//排序
	int MaxU = -100000000;
	for (int i = 1; i <= num; i++) {
		if (MaxU < ChessUtility[i].utility) {
			MaxU = ChessUtility[i].utility;
			if (Recursion == 0) {//AI设身处地下这步棋，决定最终的坐标
				ChoosePosition.x = ChessUtility[i].x;
				ChoosePosition.y = ChessUtility[i].y;
			}
		}
	}
	return	MaxU;//针对递归需要,返回最大的效用，用于上一层的计算
}

//问题有时候下棋出错，并没有翻转。有时候能下的位置下不了棋，有时候翻转了不应该翻转的棋，提示会覆盖已经下过的棋


/*int ablei[65], ablej[65];
/*int getrand(int minn, int maxn) {
	return (rand() % (maxn - minn + 1)) + minn;
}
/*position randomai(int colour) {//先胡下
	position ranpos;
	int k = 0;
	finEX = 0;
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			if (ChessBoard[i][j] == 0 && judge(i, j, colour)) {
				ablei[k] = i, ablej[k++] = j;
				finEX++;
			}
		}
	}
	if (finEX == 0) {//不给AI下棋的机会
		ranpos = { 9,9 };
		return ranpos;
	}
	srand(time(0));
	int r = getrand(0, k - 1);
	ChessBoard[ablei[r]][ablej[r]] = colour;
	allchess[ablei[r]][ablej[r]].state = 2;
	ranpos.x = ablei[r]; ranpos.y = ablej[r];
	return ranpos;
}*/

void FindPP(int color) {//找到颜色为color一方可以落子的位置；
	int num = 0;
	//ButtonSleep();
	finEX = 0;
	for (int i = 0; i < MAX_NUM; i++) {
		for (int j = 0; j < MAX_NUM; j++) {
			if (ChessBoard[i][j] != 1 && ChessBoard[i][j] != -1) {//这些位置没有棋子
				if (judge(i, j, color)) {//这个位置color可以落子
					finEX++;
					allchess[i][j].state = 1;//激活按钮
				}
			}
		}
	}
	return;
}

bool judgerev(int x, int y, int c, int Direction) {//x,y为落子的坐标，color为被翻转棋子的颜色
	if (x < 0 || x >= 8 || y < 0 || y >= 8) { return 0; }//超出边界
	if (ChessBoard[x][y] == 0) { return 0; }//遇到空位
	if (ChessBoard[x][y] == -c) { return 1; }//被夹住，但是也可能有一种特殊情况两个棋连在一起，并没有夹子
	else {
		if (judgerev(x + directx[Direction], y + directy[Direction], c, Direction)) {
			turncolor.WillTurn[turncolor.num].x = x;
			turncolor.WillTurn[turncolor.num].y = y;
			turncolor.num++;
			return 1;
		}
		return 0;
	}
}
//position corre()核心算法，机器人选择落子位置；

TurnColor JudjeWSTurn(int Colors, int x, int y) {//Agent确定要下的棋之后，或者对手落子之后，根据静态棋盘判断哪些棋将要被翻转
	for (int i = 0; i < 25; i++) {//color为需要被翻转的棋子颜色
		turncolor.WillTurn[i].x = 0;
		turncolor.WillTurn[i].y = 0;//初始化
	}
	turncolor.num = 0;//决定一共需要翻转几个棋子
	for (int i = 0; i < 8; i++) {//一共有八个方向需要判断
		int x1 = x + directx[i], y1 = y + directy[i];//某一个方向延伸
		judgerev(x1, y1, Colors, i); //这个位置以color为颜色的棋将被翻转，i代表方向
	}
	return turncolor;
}

void ReverseChess(struct TurnColor turncolor) {//接收JudgeWSTurn的返回值之后，执行翻转棋子的操作
	for (int i = 0; i < turncolor.num; i++) {
		//cout << turncolor.num << endl;
		ChessBoard[turncolor.WillTurn[i].x][turncolor.WillTurn[i].y] = -ChessBoard[turncolor.WillTurn[i].x][turncolor.WillTurn[i].y];//改变棋子颜色，之后画图就好
		//cout << ChessBoard[turncolor.WillTurn[i].x][turncolor.WillTurn[i].y]<<" "<< turncolor.WillTurn[i].x <<" "<< turncolor.WillTurn[i].y << endl;
	}
	return;
}
enum Color {//枚举可能出现的颜色
	bk = RGB(240, 135, 132),//背景颜色
	WC = RGB(255, 255, 255),//白
	BC = RGB(0, 0, 0),//黑
	BOA = RGB(237, 179, 96),//棋盘颜色
	lb = RGB(36, 199, 232),//light
	re = RGB(237, 28, 36),//red
	db = RGB(126, 132, 247),//deep blue
	ye = RGB(255, 253, 101),//yellow
};
position pos[MAX_NUM][MAX_NUM];//非指针
void GameInit() {//初始化格子左上角的坐标
	for (int i = 0; i < MAX_NUM; i++)
	{
		for (int j = 0; j < MAX_NUM; j++) {
			pos[i][j].x = (j * small_SIZE) + out_SIZE + ((j + 1) * Thin_WIDE);
			pos[i][j].y = (i * small_SIZE) + out_SIZE + ((i + 1) * Thin_WIDE);
		}
	}
}
void GameDrw() {
	//设置背景颜色
	setbkcolor(Color::bk);
	cleardevice();
	for (int i = 0; i < MAX_NUM; i++) {
		for (int j = 0; j < MAX_NUM; j++) {
			setfillcolor(Color::BOA);
			fillrectangle(pos[i][j].x, pos[i][j].y, pos[i][j].x + small_SIZE, pos[i][j].y + small_SIZE);
		}
	}
	for (int i = 0; i < MAX_NUM + 1; i++) {
		setfillcolor(Color::BC);
		fillrectangle(i * (small_SIZE + Thin_WIDE) + out_SIZE, out_SIZE, out_SIZE + i * (small_SIZE + Thin_WIDE) + Thin_WIDE, MAX_SIZE - out_SIZE);
	}
	for (int i = 0; i < MAX_NUM + 1; i++) {
		setfillcolor(Color::BC);
		fillrectangle(out_SIZE, i * (small_SIZE + Thin_WIDE) + out_SIZE, MAX_SIZE - out_SIZE, out_SIZE + i * (small_SIZE + Thin_WIDE) + Thin_WIDE);
	}

}
void ChangeColor(int ChessBoard[MAX_NUM][MAX_NUM]) {//画出棋子，这里需要强调，按钮仅提供点击和高亮服务，真正落子和改变颜色需要通过画圆
	//setfillcolor(Color::BC);
	//fillcircle(40, MAX_SIZE + 70, 50);// 此处用于检验图层顺序
	for (int i = 0; i < MAX_NUM; i++) {
		for (int j = 0; j < MAX_NUM; j++) {
			if (ChessBoard[i][j] == 1) {
				setfillcolor(Color::BC);
				fillcircle(pos[i][j].x + small_SIZE / 2, pos[i][j].y + small_SIZE / 2, 26);//请问为什么不行
				//fillcircle(out_SIZE + i * small_SIZE + (i + 1) * Thin_WIDE + small_SIZE / 2, out_SIZE + j * small_SIZE + (j + 1) * Thin_WIDE + small_SIZE / 2, 26);
			}
			if (ChessBoard[i][j] == -1) {
				setfillcolor(Color::WC);
				fillcircle(pos[i][j].x + small_SIZE / 2, pos[i][j].y + small_SIZE / 2, 26);
				//fillcircle(out_SIZE + i * small_SIZE + (i + 1) * Thin_WIDE + small_SIZE / 2, out_SIZE + j * small_SIZE + (j + 1) * Thin_WIDE + small_SIZE / 2, 26);
			}
		}
	}
}

void initializebutton() {//先定义一个全局变量然后在调用函数时初始化按钮
	for (int i = 0; i < MAX_NUM; i++) {
		for (int j = 0; j < MAX_NUM; j++) {
			allchess[i][j] = { pos[i][j].x + small_SIZE / 2, pos[i][j].y + small_SIZE / 2, 26, BOA, ye, re, BOA, 0 };
			drawChessButton(allchess[i][j]);//0为休眠态，不作提示，人类无法落子，1为激活态，可以提示，人类可以落子。
		}
	}
}
void drawChessButton(struct ChessButton theButton) {
	setfillcolor(theButton.curcolor);
	solidcircle(theButton.x, theButton.y, 26);
	//setbkmode(TRANSPARENT);
}
struct Button {
	int x;
	int y;
	int w;
	int h;
	COLORREF curColor;//当前颜色
	COLORREF inColor;//鼠标在里面的颜色
	COLORREF outColor;//鼠标不在
	const char* str;
	COLORREF textColor;
};
struct Button White = { 40,MAX_SIZE + 70,75,40,WC,re,WC,"White",db };
struct	Button Black = { 160,MAX_SIZE + 70,75,40, BC, re, BC,"Black",WC };
struct	Button Tips = { 290,MAX_SIZE + 70,75,40,lb,re,lb ,"Tips",WC };
struct	Button Giveup = { 400,MAX_SIZE + 70,75,40, ye, re,ye,"Resign",db };
struct  Button Infor = { 80,MAX_SIZE + 10,400,50,WC, WC,WC, "Please choose your color.",BC };//通知窗口
struct Button save = { 130,MAX_SIZE + 140,80,40,WC, re,WC, "Save",BC };
struct Button  read= { 300,MAX_SIZE + 140,80,40,WC, re,WC, "Read",BC };
void drawButton(struct Button button) {
	//draw a rectangle;
	setfillcolor(button.curColor);
	solidrectangle(button.x, button.y, button.x + button.w, button.y + button.h);
	//把文字居中；
	setbkmode(TRANSPARENT);
	settextcolor(button.textColor);
	settextstyle(button.h * 6/11, 0, "黑体");
	int w = (button.w - textwidth(button.str)) / 2;
	int h = (button.h - textheight(button.str)) / 2;
	outtextxy(button.x + w, button.y + h, button.str);
}
struct Supervise {
	int x;
	int y;
	int state;//state为1代表人类已经落子，可以进行后续操作，为0则继续等待,如果state为2，无处可走
};

void ButtonSleep() {//将激活按钮恢复，
	for (int i = 0; i < MAX_NUM; i++) {
		for (int j = 0; j < MAX_NUM; j++) {
			if (ChessBoard[i][j] != 1 && ChessBoard[i][j] != -1 && allchess[i][j].state != 2) {//没有下棋的按钮，恢复隐藏状态，这些棋子下一步可能不能下了
				allchess[i][j].state = 0;
				UnHighlight(&allchess[i][j]);
				drawChessButton(allchess[i][j]);
			}
		}
	}
	return;
}
void UnHighlight(ChessButton* chess) {
	chess->curcolor = chess->hidecolor;
}
void setAllchess() {//将不可下棋的位置屏蔽
	for (int i = 0; i < MAX_NUM; i++) {
		for (int j = 0; j < MAX_NUM; j++) {
			if (ChessBoard[i][j] != 0) {
				allchess[i][j].state = 2;// when read,change the chessbuttons' state
			}
		}
	}
}
void readboard(int board[8][8]) {//读取棋盘
	ifstream infile("board.txt");
	if (!infile) {
		cerr << "open outfile error" << endl;
		exit(1);
	}
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 8; j++)
			infile >> board[i][j];
	infile.close();
	return;
}
void readcolour(int c[1]) {//读取颜色
	ifstream infile("colour.txt");
	if (!infile) {
		cerr << "open outfile error" << endl;
		exit(1);
	}
	infile >> c[0];
	return;
}
void saveboard(int board[8][8]) {//以数组的方式，储存棋盘
	ofstream outfile("board.txt");
	if (!outfile) {
		cerr << "open outfile error" << endl;
		exit(1);
	}
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 8; j++)
			outfile << board[i][j] << endl;
	outfile.close();
	return;
}
void savecolour(int c[1]) {//以一个元素的数组存下棋者的颜色
	ofstream outfile("colour.txt");
	if (!outfile) {
		cerr << "open outfile error" << endl;
		exit(1);
	}
	outfile << c[0] << endl;
	outfile.close();
	return;
}

Supervise JWHHP(int color) {//等待，直到人类落子,并返回人类的落子坐标。//color为人类的棋的颜色，则判断人类能下哪几步

	ExMessage msg;
	Supervise result;
	while (1) {//最初按钮不变色的原因是循环无法一直进行下去，没有Drawbutton
		peekmessage(&msg, EM_MOUSE);
		//!!
		FindPP(color);//color棋可以落子的位置，将这些按钮激活
		if (!finEX) {//无棋可下
			result = { 0,0,2 };//2是异常情况
			return result;
		}
		//!!
		drawButton(Giveup);
		drawButton(Tips);
		drawButton(read);
		drawButton(save);
		for (int i = 0; i < MAX_NUM; i++) {
			for (int j = 0; j < MAX_NUM; j++) {//state为1，按钮被激活
				if (allchess[i][j].state == 1) { //如果按钮被激活了，将按钮高亮，并画出来
					drawChessButton(allchess[i][j]); //画上按钮
				}
			}
		}
		if (clickButton(&Tips, msg)) {//为接下来人类落子提供选点提示
			for (int i = 0; i < MAX_NUM; i++) {
				for (int j = 0; j < MAX_NUM; j++) {//state为1，按钮被激活
					if (allchess[i][j].state == 1) { //如果按钮被激活了，将按钮高亮，并画出来
						HighlightTip(&allchess[i][j]); //按钮变色
						drawChessButton(allchess[i][j]); //画上按钮
					}
				}
			}
		}
		if (clickButton(&Giveup, msg)) //点击投降，中途好像不太行
		{
			WhoseTurn(&Infor, 1); drawButton(Infor);
			result = { 1,1,10 };
			return result;
		}
		if (clickButton(&save, msg)) {//
			int c[1]{};
			c[0] = color;
			savecolour(c);
			saveboard(ChessBoard);
		}
		for (int i = 0; i < MAX_NUM; i++) {
			for (int j = 0; j < MAX_NUM; j++) {
				if (clickChessButton(&allchess[i][j], msg) && allchess[i][j].state == 1) {
					allchess[i][j].state = 2;//之后不可落子，改变按钮棋盘
					result = { i,j,1 };
					return result;
				}
			}
		}
		FlushBatchDraw();//循环可以一直走到这一步，让按钮不在屏闪
	}
}
//鼠标在按钮上
void WhoseTurn(Button* Infor, int information) {//显示对话信息
	if (information == 1) {
		Infor->str = "This your turn.";
	}
	else if (information == 2) {
		Infor->str = "Please wait a second.";
	}
	else if (information == 0) {
		Infor->str = "很遗憾，你输了";
	}
	else if (information == -1) {
		Infor->str = "恭喜，你赢了";
	}
	else if (information == 3) {
		Infor->str = "你和AI和棋了";
	}
}
void HighlightTip(ChessButton* chess) {
	chess->curcolor = chess->choicecolor;
}
int mouseInButton(struct Button* button, ExMessage msg)//判断鼠标的位置
{
	if (msg.x >= button->x && msg.y >= button->y &&
		msg.x <= button->x + button->w && msg.y <= button->y + button->h)
	{
		button->curColor = button->inColor;
		return 1;
	}
	button->curColor = button->outColor;
	return 0;
}
//点击鼠标
int clickButton(struct Button* button, ExMessage msg)
{
	if (mouseInButton(button, msg) && msg.message == WM_LBUTTONDOWN) {
		return 1;
	}
	else { return 0; }
}
int mouseInChessButton(struct ChessButton* button, ExMessage msg)//判断鼠标的位置
{
	if (msg.x >= button->x - button->r && msg.y >= button->y - button->r &&
		msg.x <= button->x + button->r && msg.y <= button->y + button->r)
	{
		button->curcolor = button->incolor;
		return 1;
	}
	button->curcolor = button->hidecolor;
	return 0;
}
//点击鼠标
int clickChessButton(struct ChessButton* chessbutton, ExMessage msg)
{
	if (mouseInChessButton(chessbutton, msg) && msg.message == WM_LBUTTONDOWN) {
		return 1;
	}
	else { return 0; }
}
int Choosecolor = 0;//机器人的颜色，如果机器将要下棋则结果为1，机器落完子之后结果为-1
int ChessState = 0;//棋局形势，该AI下则为1。  该人类则为0，在这个阶段：人类可以请求提示，可以弃权
bool IFend() {
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			cout << ChessBoard[i][j] << "\t";
		}
		cout << endl;
		cout << endl;
	}
	for (int i = 0; i < MAX_NUM; i++) {
		for (int j = 0; j < MAX_NUM; j++) {
			if (ChessBoard[i][j] == 0) {
				return 0;
			}
		}
	}
	//cout << "???" << endl;
	return 1;//已经全部别占满
}
int JUDGEresult() {//白棋获胜返回-1，黑棋获胜返回1；平局返回0
	int sum = 0;
	for (int i = 0; i < MAX_NUM; i++) {
		for (int j = 0; j < MAX_NUM; j++) {
			sum += ChessBoard[i][j];
		}
	}
	if (sum < 0) {
		return -1;//白胜
	}
	else if (sum > 0) {
		return 1;//黑胜
	}
	else {
		return 0;//平局
	}
}
int WINRE = 0;//统计最终哪一方的棋子更多
int numOfStep = 0;//count the num of steps
int main() {
	fstream myFile("board.txt", ios::in | ios::out), myFile2("colour.txt", ios::in | ios::out);
	initgraph(MAX_SIZE, MAX_SIZE + 210);
	GameInit();
	GameDrw();
	ExMessage msg;
	ChessBoard[3][3] = -1; ChessBoard[4][4] = -1;
	ChessBoard[3][4] = 1; ChessBoard[4][3] = 1;
	drawButton(Infor);
	initializebutton();
	//BeginBatchDraw();
	peekmessage(&msg, EM_MOUSE);
	while (1) {//绘制棋盘，各个按钮，实现选择黑白,如果选择了黑白，就跳出循环
		drawButton(White);
		drawButton(Black);//即使不画按钮，点击鼠标也能起作用
		drawButton(Giveup);
		drawButton(Tips);
		drawButton(read);
		drawButton(save);
		ChangeColor(ChessBoard);
		peekmessage(&msg, EM_MOUSE);
		if (clickButton(&White, msg)) { Choosecolor = 1; ChessState = 1; break; }//if chessstate=1,it's AI's turn
		if (clickButton(&Black, msg)) { Choosecolor = -1; ChessState = 0; WhoseTurn(&Infor, 1); drawButton(Infor); break; }
		if (clickButton(&Giveup, msg)) { WhoseTurn(&Infor, 0); drawButton(Infor); }
		if (clickButton(&read, msg)) {
			int rc[1] = { 1 };
			readcolour(rc);//rc save the humman's color
			Choosecolor = -rc[0];
			readboard(ChessBoard);
			setAllchess();
			ChangeColor(ChessBoard);
			ChessState = 0;
			break;
		};
		FlushBatchDraw();
	}
	//EndBatchDraw();
	//BeginBatchDraw();
	//peekmessage(&msg, EM_MOUSE);
	while (1) {
		peekmessage(&msg, EM_MOUSE);
		//drawButton(Giveup);
		//drawButton(Tips);
		ChangeColor(ChessBoard);
		if (ChessState) {//该机器下棋//机器找到落子选点;
			WhoseTurn(&Infor, 2);
			drawButton(Infor);
			int curTime = clock();
			FindAI(Choosecolor);//计算落子位置
			while (clock() - curTime < CLOCKS_PER_SEC);
			position AIpos = ChoosePosition;//落子，并对两种棋盘作出更改
			numOfStep ++;//a new num that count the num of chess
			if (AIpos.x == 9) {
				ChessState = 0;
				if (IFend()) {
					goto finalJUD;
				}
				else { goto Hummanshow; }
			}
			allchess[AIpos.x][AIpos.y].state = 2;//决定好走这步之后改变按钮状态
			ChessBoard[AIpos.x][AIpos.y] = Choosecolor;//确定落子

			ChangeColor(ChessBoard);
			Sleep(40);
			cout << AIpos.x << " " << AIpos.y << endl;
			for (int i = 0; i < 8; i++) {
				for (int j = 0; j < 8; j++) {
					cout << ChessBoard[i][j] << "\t";
				}
				cout << endl;
				cout << endl;
			}
			ReverseChess(JudjeWSTurn(-Choosecolor, AIpos.x, AIpos.y));//翻转人类的棋
			//Sleep(2000);
			ChangeColor(ChessBoard);
			ChessState = 0;
			FlushBatchDraw();
			//ChessBoard[][] = -Choosecolor;
		}
	Hummanshow:	while (!ChessState) {//等待人类落子，并获取落子位置
		WhoseTurn(&Infor, 1);
		drawButton(Infor);
		if (numOfStep == 20 || numOfStep ==21 || numOfStep == 22) { changeArr(); }
		if (numOfStep == 55 || numOfStep == 55 || numOfStep == 55) { changeArrAgain(); }
		Supervise Information = JWHHP(-Choosecolor);
		numOfStep++;//a new num that count the num of chess
		if (Information.state == 1) {//人类已经落子,就可以进行翻转了
			ChessState = 1;//下一步智能下棋
		}//改变数据棋盘
		else if (Information.state == 2) {//不给人类下棋的机会
			ChessState = 1;//该AI下棋
			if (IFend()) {//如果结束了
				//cout << "COMEIN" << endl;
				goto finalJUD;
			}
			break;
		}
		if (Information.state == 10) {//人类弃权
			WhoseTurn(&Infor, 0); drawButton(Infor);
			Sleep(5000);
			return 0;
		}
		ChessBoard[Information.x][Information.y] = -Choosecolor;//这个棋盘上的位置最先被人类下了；翻转颜色为Choosecolor的棋子
		allchess[Information.x][Information.y].state = 2;
		ButtonSleep();
		ReverseChess(JudjeWSTurn(Choosecolor, Information.x, Information.y));//翻转机器人的棋
		Sleep(100);
		ChangeColor(ChessBoard);
		//FlushBatchDraw();
	}//之后就轮到智能下棋了
	FlushBatchDraw();
	}
finalJUD:	WINRE = JUDGEresult();
	if (WINRE * Choosecolor == 0) {
		WhoseTurn(&Infor, 3);
		drawButton(Infor);
		Sleep(5000);
		return 0;
	}
	else if (WINRE * Choosecolor > 0) {
		WhoseTurn(&Infor, 0);
		drawButton(Infor);
		Sleep(5000);
		return 0;
	}
	else {
		WhoseTurn(&Infor, -1);
		drawButton(Infor);
		Sleep(5000);
		return 0;
	}
	while (1) {};
	EndBatchDraw();//双缓冲
	return 0;
}
//我的目的，从3步开始，尽量四步合理利用剪枝，将棋盘的每个棋子做成按钮结构体，没下的时候可以起按钮的作用，下过之后只改变颜色
//鼠标在上方高亮，提示可以下棋时高亮.
//难点：保存数组，如何在搜索时不改变元素？
//写一个接口，点击按钮传入，将翻转的棋的位置传出，将可下位置传出，交互窗口改变棋的颜色，并且高亮对方可以下棋的位置