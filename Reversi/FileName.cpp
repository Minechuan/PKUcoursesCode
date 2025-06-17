#include<iostream>
#include<graphics.h>
#include<cstdlib>
#include<random>
#include<time.h>
#include<fstream>
using namespace std;
#define MAX_NUM 8//����
#define MAX_SIZE 567//���̿��
#define Thin_WIDE 7//�������
#define small_SIZE 60//���߳�
#define out_SIZE 12//�߿���
//240,155,89
struct position {
	int x;
	int y;
};
struct ChessButton {
	int x;
	int y;
	int r;
	COLORREF hidecolor;//����Ϊ�����̵���ɫһ��
	COLORREF choicecolor;//��ʾ�����ߣ�����������ӵ�λ��
	COLORREF incolor;//������������ɫ
	COLORREF curcolor;//����ط�Ŀǰ��
	int state;//1 �����0 �������ߣ�2�����Ѿ�������֮���޷����ӡ�
};
struct EACHCHESS {
	int x;
	int y;
	int utility;//������Ч��
};
//һ����Ҫ�Ľṹ�壬���������ݹ�ͻ���
struct TurnColor {//���վ����ý�Ҫ��ת�ĵ�󣬴�������ṹ��
	position WillTurn[40];//�ж���Ҫ��ת�ĵ㣬
	int num;//һ���м��������ĵ�
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
#define MAX_NUM 8//������
bool judge(int x, int y, int color); //�ж��Ƿ����,��ɫΪcolor������
int ChessBoard[MAX_NUM][MAX_NUM]{};//����Ϊ1������Ϊ-1��û�������Ϊ0��

const int directx[8] = { 1,0,-1, 0,1, 1,-1,-1 };//�����뷽��
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
void changeArr() {//15��֮���Ȩ������
	for (int i = 0; i < MAX_NUM; i++) {
		for (int j = 0; j < MAX_NUM; j++) {
			arr[i][j] = arrLater[i][j];
		}
	}
	return;
}
void changeArrAgain() {//��ֽӽ�β����Ȩ������
	for (int i = 0; i < MAX_NUM; i++) {
		for (int j = 0; j < MAX_NUM; j++) {
			arr[i][j] = arrFinal[i][j];
		}
	}
	return;
}
//Ȩֵ����ʹ��Ȩֵ��
//struct SOMEdate {//Ч��Ϊ�������ߵ��������
// Part;//Ȩֵ
// possibility;//��һ���Է����ж���
//Ч��+�Է�����Ч��=�����׼��
//�������ܻ��γ����ޣ�Ϊ�˵ݹ��ܹ�return ��������󣬰ѵ��岽�����С�ö�Ϊ0,
//���ʱ�����

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
bool AIrev(int x, int y, int color, int Direction, TurnColor& AIturn) {//x,yΪ���ӵ����꣬colorΪ����ת���ӵ���ɫ
	if (x < 0 || x >= 8 || y < 0 || y >= 8) { return 0; }//�����߽�
	if (ChessBoard[x][y] == 0) { return 0; }//������λ
	if (ChessBoard[x][y] == -color) { return 1; }//����ס������Ҳ������һ�������������������һ�𣬲�û�м���
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
TurnColor AISTurn(int Colors, int x, int y, TurnColor& AIturn) {//���ݾ�̬�����ж���Щ�彫Ҫ����ת
	for (int i = 0; i < 25; i++) {//colorΪ��Ҫ����ת��������ɫ
		AIturn.WillTurn[i].x = 0;
		AIturn.WillTurn[i].y = 0;//��ʼ��
	}
	AIturn.num = 0;//����һ����Ҫ��ת��������
	for (int i = 0; i < 8; i++) {//һ���а˸�������Ҫ�ж�
		int x1 = x + directx[i], y1 = y + directy[i];//ĳһ����������
		AIrev(x1, y1, Colors, i, AIturn); //���λ����colorΪ��ɫ���彫����ת��i������
	}
	return AIturn;
}

int MoveMent(int color) {//����ͳ���ж�����������0
	int MOVEnum = 0;
	//EACHCHESS ChessUtility[20]{ };//�������浱ǰ����µĿ��е�
	for (int i = 0; i < MAX_NUM; i++) {
		for (int j = 0; j < MAX_NUM; j++) {
			if (ChessBoard[i][j] != 1 && ChessBoard[i][j] != -1 && allchess[i][j].state != 2) {//��Щλ��û������
				if (judge(i, j, color)) {//���λ��color��������
					MOVEnum++;
				}
			}
		}
	}
	return MOVEnum;
}

bool judge(int x, int y, int color) {//�ж��Ƿ����,��ɫΪcolor������
	int i;//�˸�����
	for (i = 0; i < 8; i++) {//8������̽���Ƿ����ӿɳ�
		bool gap = 0;//�Ƿ��м��
		int x1 = x + directx[i], y1 = y + directy[i];// , tempchange = 0;
		while (1) {
			if (x1 < 0 || x1 >= 8 || y1 < 0 || y1 >= 8) { break; }//����ֱ��return�������޷�����8������
			if (ChessBoard[x1][y1] == 0)//���ǿ�λ
				break;
			else if (ChessBoard[x1][y1] == color) {//������Լ�����ɫ
				if (gap) { return gap; }
				else { break; }
			}
			else {//�ǶԷ�����//���ŶԷ�����ɫ
				gap = 1;
				x1 += directx[i];
				y1 += directy[i];
			}
		}
	}
	return 0;//pieceschange;
}
position ChoosePosition;
int Recursion = 0;//ͳ�Ƶݹ��������������
bool comPare(int *a,int *b) {//���Ϊtrue�ͼ�ȥ
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
		if (*a > 0.6 * *b) {//��Ҫ��
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
	else {//���ܼ�
		*a = *b;
		return false;
	}
}
int FindAI(int color) {//�ҵ���ɫΪcolor�������ӵ�λ�ã���ʼΪAI����ɫ
	int finAI = 0; //�ж��Ƿ��������
	int num = 0;//��һ��ֵ����1��
	EACHCHESS ChessUtility[40]{ };//��ǰ���������Ҫ��������
	for (int i = 0; i < MAX_NUM; i++) {//�������̣�Ѱ�ҿ������ӵĵط�
		for (int j = 0; j < MAX_NUM; j++) {
			if (ChessBoard[i][j] != 1 && ChessBoard[i][j] != -1) {//��Щλ��û������
				if (judge(i, j, color)) {//���λ��color��������
					num++;//��ѡ��λ�ø���
					finAI++;//�������
					int curWeight = CALcu();//��ǰȨ�غ�

					ChessBoard[i][j] = color;//ģ�������ⲽ��
					allchess[i][j].state = 2;
					Recursion++;//���ݹ����
					//ģ�ⷭת��Ȼ��ݹ�
					TurnColor AIturn;//����ṹ�壬���ڴ��潫Ҫ��ת����
					AISTurn(-color, i, j, AIturn);//colorΪ�����ߵ���ɫ��-colorΪ����ת�ߵ���ɫ
					//�õ�����ת���������꼰����
					for (int imire = 0; imire < AIturn.num; imire++) {//���и�ֵ
						ChessBoard[AIturn.WillTurn[imire].x][AIturn.WillTurn[imire].y] = color;//����ɫΪcolor��һ����ת
					}
					//������һ��֮�����̵�Ȩ��;
					int laterWeight = CALcu();
					if (Recursion == 4) {//�㵽k��֮�󲻵���FindAI�Ͳ�����
						//cout << "five steps" << endl;//���û����£��Ͳ�������,�������֮���������
						ChessUtility[num].utility = 30*(laterWeight - curWeight) -  90* MoveMent(-color);
					}
					else { 
						
						int compareFirst = 30* (laterWeight - curWeight) - 80 * MoveMent(-color);//Ϊ�˼�֦����ǰ�жϾֲ�Ч�ú���
						if (i == 0 && j == 0 || i == 7 && j == 0 || i == 0 && j == 7 || i == 7 && j == 7) {
							compareFirst += 100000;
							cout << compareFirst << " ";
							//Sleep(100);
						}
						int MAXUUU = 0;//Ŀǰ��ǰ4���������Ч��
						if (num < 3&&Recursion>=2&&compareFirst>MAXUUU&& Recursion >= 2) {//�жϼ�֦����
							MAXUUU = compareFirst;//
						}
						if (num >= 3 && Recursion >= 2 && comPare(&MAXUUU,&compareFirst)){//MAXUUUΪ��ʱ�����ֵ���ܻ����
							num--;//the position and it's later result will not be take into account;
							goto StopThinking;//�ݹ����Ϊ��֮��ʼ��֦�����С̫�࣬��������
						}
						ChessUtility[num].x = i;
						ChessUtility[num].y = j;
						ChessUtility[num].utility =compareFirst-10*FindAI(-color);
						          //Ȩ�ز���������������������Է��ж������������������������Է����Ų���Ч��
					}
					cout << ChessUtility[num].utility << endl;
					//Sleep(100);
					//�����跭ת���巭ת����
StopThinking:		for (int imire = 0; imire < AIturn.num; imire++) {//���и�ֵ
						ChessBoard[AIturn.WillTurn[imire].x][AIturn.WillTurn[imire].y] = -color;
					}//����ɫΪcolor��һ��������
					Recursion--;//��ȳ���1
					ChessBoard[i][j] = 0;//���ݣ�ԭ��û����
					allchess[i][j].state = 0;
				}
			}
		}
	}
	if (!finAI) {//FinAI�����ȫ�ֱ����Ͳ��ô���
		//cout << "fehoijfwi" << endl;
		ChoosePosition.x = 9;
		ChoosePosition.y = 9;
		return 0;//һ��������¡�����ֱ�Ӽ����������
	}
	//����
	int MaxU = -100000000;
	for (int i = 1; i <= num; i++) {
		if (MaxU < ChessUtility[i].utility) {
			MaxU = ChessUtility[i].utility;
			if (Recursion == 0) {//AI���������ⲽ�壬�������յ�����
				ChoosePosition.x = ChessUtility[i].x;
				ChoosePosition.y = ChessUtility[i].y;
			}
		}
	}
	return	MaxU;//��Եݹ���Ҫ,��������Ч�ã�������һ��ļ���
}

//������ʱ�����������û�з�ת����ʱ�����µ�λ���²����壬��ʱ��ת�˲�Ӧ�÷�ת���壬��ʾ�Ḳ���Ѿ��¹�����


/*int ablei[65], ablej[65];
/*int getrand(int minn, int maxn) {
	return (rand() % (maxn - minn + 1)) + minn;
}
/*position randomai(int colour) {//�Ⱥ���
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
	if (finEX == 0) {//����AI����Ļ���
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

void FindPP(int color) {//�ҵ���ɫΪcolorһ���������ӵ�λ�ã�
	int num = 0;
	//ButtonSleep();
	finEX = 0;
	for (int i = 0; i < MAX_NUM; i++) {
		for (int j = 0; j < MAX_NUM; j++) {
			if (ChessBoard[i][j] != 1 && ChessBoard[i][j] != -1) {//��Щλ��û������
				if (judge(i, j, color)) {//���λ��color��������
					finEX++;
					allchess[i][j].state = 1;//���ť
				}
			}
		}
	}
	return;
}

bool judgerev(int x, int y, int c, int Direction) {//x,yΪ���ӵ����꣬colorΪ����ת���ӵ���ɫ
	if (x < 0 || x >= 8 || y < 0 || y >= 8) { return 0; }//�����߽�
	if (ChessBoard[x][y] == 0) { return 0; }//������λ
	if (ChessBoard[x][y] == -c) { return 1; }//����ס������Ҳ������һ�������������������һ�𣬲�û�м���
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
//position corre()�����㷨��������ѡ������λ�ã�

TurnColor JudjeWSTurn(int Colors, int x, int y) {//Agentȷ��Ҫ�µ���֮�󣬻��߶�������֮�󣬸��ݾ�̬�����ж���Щ�彫Ҫ����ת
	for (int i = 0; i < 25; i++) {//colorΪ��Ҫ����ת��������ɫ
		turncolor.WillTurn[i].x = 0;
		turncolor.WillTurn[i].y = 0;//��ʼ��
	}
	turncolor.num = 0;//����һ����Ҫ��ת��������
	for (int i = 0; i < 8; i++) {//һ���а˸�������Ҫ�ж�
		int x1 = x + directx[i], y1 = y + directy[i];//ĳһ����������
		judgerev(x1, y1, Colors, i); //���λ����colorΪ��ɫ���彫����ת��i������
	}
	return turncolor;
}

void ReverseChess(struct TurnColor turncolor) {//����JudgeWSTurn�ķ���ֵ֮��ִ�з�ת���ӵĲ���
	for (int i = 0; i < turncolor.num; i++) {
		//cout << turncolor.num << endl;
		ChessBoard[turncolor.WillTurn[i].x][turncolor.WillTurn[i].y] = -ChessBoard[turncolor.WillTurn[i].x][turncolor.WillTurn[i].y];//�ı�������ɫ��֮��ͼ�ͺ�
		//cout << ChessBoard[turncolor.WillTurn[i].x][turncolor.WillTurn[i].y]<<" "<< turncolor.WillTurn[i].x <<" "<< turncolor.WillTurn[i].y << endl;
	}
	return;
}
enum Color {//ö�ٿ��ܳ��ֵ���ɫ
	bk = RGB(240, 135, 132),//������ɫ
	WC = RGB(255, 255, 255),//��
	BC = RGB(0, 0, 0),//��
	BOA = RGB(237, 179, 96),//������ɫ
	lb = RGB(36, 199, 232),//light
	re = RGB(237, 28, 36),//red
	db = RGB(126, 132, 247),//deep blue
	ye = RGB(255, 253, 101),//yellow
};
position pos[MAX_NUM][MAX_NUM];//��ָ��
void GameInit() {//��ʼ���������Ͻǵ�����
	for (int i = 0; i < MAX_NUM; i++)
	{
		for (int j = 0; j < MAX_NUM; j++) {
			pos[i][j].x = (j * small_SIZE) + out_SIZE + ((j + 1) * Thin_WIDE);
			pos[i][j].y = (i * small_SIZE) + out_SIZE + ((i + 1) * Thin_WIDE);
		}
	}
}
void GameDrw() {
	//���ñ�����ɫ
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
void ChangeColor(int ChessBoard[MAX_NUM][MAX_NUM]) {//�������ӣ�������Ҫǿ������ť���ṩ����͸��������������Ӻ͸ı���ɫ��Ҫͨ����Բ
	//setfillcolor(Color::BC);
	//fillcircle(40, MAX_SIZE + 70, 50);// �˴����ڼ���ͼ��˳��
	for (int i = 0; i < MAX_NUM; i++) {
		for (int j = 0; j < MAX_NUM; j++) {
			if (ChessBoard[i][j] == 1) {
				setfillcolor(Color::BC);
				fillcircle(pos[i][j].x + small_SIZE / 2, pos[i][j].y + small_SIZE / 2, 26);//����Ϊʲô����
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

void initializebutton() {//�ȶ���һ��ȫ�ֱ���Ȼ���ڵ��ú���ʱ��ʼ����ť
	for (int i = 0; i < MAX_NUM; i++) {
		for (int j = 0; j < MAX_NUM; j++) {
			allchess[i][j] = { pos[i][j].x + small_SIZE / 2, pos[i][j].y + small_SIZE / 2, 26, BOA, ye, re, BOA, 0 };
			drawChessButton(allchess[i][j]);//0Ϊ����̬��������ʾ�������޷����ӣ�1Ϊ����̬��������ʾ������������ӡ�
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
	COLORREF curColor;//��ǰ��ɫ
	COLORREF inColor;//������������ɫ
	COLORREF outColor;//��겻��
	const char* str;
	COLORREF textColor;
};
struct Button White = { 40,MAX_SIZE + 70,75,40,WC,re,WC,"White",db };
struct	Button Black = { 160,MAX_SIZE + 70,75,40, BC, re, BC,"Black",WC };
struct	Button Tips = { 290,MAX_SIZE + 70,75,40,lb,re,lb ,"Tips",WC };
struct	Button Giveup = { 400,MAX_SIZE + 70,75,40, ye, re,ye,"Resign",db };
struct  Button Infor = { 80,MAX_SIZE + 10,400,50,WC, WC,WC, "Please choose your color.",BC };//֪ͨ����
struct Button save = { 130,MAX_SIZE + 140,80,40,WC, re,WC, "Save",BC };
struct Button  read= { 300,MAX_SIZE + 140,80,40,WC, re,WC, "Read",BC };
void drawButton(struct Button button) {
	//draw a rectangle;
	setfillcolor(button.curColor);
	solidrectangle(button.x, button.y, button.x + button.w, button.y + button.h);
	//�����־��У�
	setbkmode(TRANSPARENT);
	settextcolor(button.textColor);
	settextstyle(button.h * 6/11, 0, "����");
	int w = (button.w - textwidth(button.str)) / 2;
	int h = (button.h - textheight(button.str)) / 2;
	outtextxy(button.x + w, button.y + h, button.str);
}
struct Supervise {
	int x;
	int y;
	int state;//stateΪ1���������Ѿ����ӣ����Խ��к���������Ϊ0������ȴ�,���stateΪ2���޴�����
};

void ButtonSleep() {//�����ť�ָ���
	for (int i = 0; i < MAX_NUM; i++) {
		for (int j = 0; j < MAX_NUM; j++) {
			if (ChessBoard[i][j] != 1 && ChessBoard[i][j] != -1 && allchess[i][j].state != 2) {//û������İ�ť���ָ�����״̬����Щ������һ�����ܲ�������
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
void setAllchess() {//�����������λ������
	for (int i = 0; i < MAX_NUM; i++) {
		for (int j = 0; j < MAX_NUM; j++) {
			if (ChessBoard[i][j] != 0) {
				allchess[i][j].state = 2;// when read,change the chessbuttons' state
			}
		}
	}
}
void readboard(int board[8][8]) {//��ȡ����
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
void readcolour(int c[1]) {//��ȡ��ɫ
	ifstream infile("colour.txt");
	if (!infile) {
		cerr << "open outfile error" << endl;
		exit(1);
	}
	infile >> c[0];
	return;
}
void saveboard(int board[8][8]) {//������ķ�ʽ����������
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
void savecolour(int c[1]) {//��һ��Ԫ�ص�����������ߵ���ɫ
	ofstream outfile("colour.txt");
	if (!outfile) {
		cerr << "open outfile error" << endl;
		exit(1);
	}
	outfile << c[0] << endl;
	outfile.close();
	return;
}

Supervise JWHHP(int color) {//�ȴ���ֱ����������,������������������ꡣ//colorΪ����������ɫ�����ж����������ļ���

	ExMessage msg;
	Supervise result;
	while (1) {//�����ť����ɫ��ԭ����ѭ���޷�һֱ������ȥ��û��Drawbutton
		peekmessage(&msg, EM_MOUSE);
		//!!
		FindPP(color);//color��������ӵ�λ�ã�����Щ��ť����
		if (!finEX) {//�������
			result = { 0,0,2 };//2���쳣���
			return result;
		}
		//!!
		drawButton(Giveup);
		drawButton(Tips);
		drawButton(read);
		drawButton(save);
		for (int i = 0; i < MAX_NUM; i++) {
			for (int j = 0; j < MAX_NUM; j++) {//stateΪ1����ť������
				if (allchess[i][j].state == 1) { //�����ť�������ˣ�����ť��������������
					drawChessButton(allchess[i][j]); //���ϰ�ť
				}
			}
		}
		if (clickButton(&Tips, msg)) {//Ϊ���������������ṩѡ����ʾ
			for (int i = 0; i < MAX_NUM; i++) {
				for (int j = 0; j < MAX_NUM; j++) {//stateΪ1����ť������
					if (allchess[i][j].state == 1) { //�����ť�������ˣ�����ť��������������
						HighlightTip(&allchess[i][j]); //��ť��ɫ
						drawChessButton(allchess[i][j]); //���ϰ�ť
					}
				}
			}
		}
		if (clickButton(&Giveup, msg)) //���Ͷ������;����̫��
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
					allchess[i][j].state = 2;//֮�󲻿����ӣ��ı䰴ť����
					result = { i,j,1 };
					return result;
				}
			}
		}
		FlushBatchDraw();//ѭ������һֱ�ߵ���һ�����ð�ť��������
	}
}
//����ڰ�ť��
void WhoseTurn(Button* Infor, int information) {//��ʾ�Ի���Ϣ
	if (information == 1) {
		Infor->str = "This your turn.";
	}
	else if (information == 2) {
		Infor->str = "Please wait a second.";
	}
	else if (information == 0) {
		Infor->str = "���ź���������";
	}
	else if (information == -1) {
		Infor->str = "��ϲ����Ӯ��";
	}
	else if (information == 3) {
		Infor->str = "���AI������";
	}
}
void HighlightTip(ChessButton* chess) {
	chess->curcolor = chess->choicecolor;
}
int mouseInButton(struct Button* button, ExMessage msg)//�ж�����λ��
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
//������
int clickButton(struct Button* button, ExMessage msg)
{
	if (mouseInButton(button, msg) && msg.message == WM_LBUTTONDOWN) {
		return 1;
	}
	else { return 0; }
}
int mouseInChessButton(struct ChessButton* button, ExMessage msg)//�ж�����λ��
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
//������
int clickChessButton(struct ChessButton* chessbutton, ExMessage msg)
{
	if (mouseInChessButton(chessbutton, msg) && msg.message == WM_LBUTTONDOWN) {
		return 1;
	}
	else { return 0; }
}
int Choosecolor = 0;//�����˵���ɫ�����������Ҫ��������Ϊ1������������֮����Ϊ-1
int ChessState = 0;//������ƣ���AI����Ϊ1��  ��������Ϊ0��������׶Σ��������������ʾ��������Ȩ
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
	return 1;//�Ѿ�ȫ����ռ��
}
int JUDGEresult() {//�����ʤ����-1�������ʤ����1��ƽ�ַ���0
	int sum = 0;
	for (int i = 0; i < MAX_NUM; i++) {
		for (int j = 0; j < MAX_NUM; j++) {
			sum += ChessBoard[i][j];
		}
	}
	if (sum < 0) {
		return -1;//��ʤ
	}
	else if (sum > 0) {
		return 1;//��ʤ
	}
	else {
		return 0;//ƽ��
	}
}
int WINRE = 0;//ͳ��������һ�������Ӹ���
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
	while (1) {//�������̣�������ť��ʵ��ѡ��ڰ�,���ѡ���˺ڰף�������ѭ��
		drawButton(White);
		drawButton(Black);//��ʹ������ť��������Ҳ��������
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
		if (ChessState) {//�û�������//�����ҵ�����ѡ��;
			WhoseTurn(&Infor, 2);
			drawButton(Infor);
			int curTime = clock();
			FindAI(Choosecolor);//��������λ��
			while (clock() - curTime < CLOCKS_PER_SEC);
			position AIpos = ChoosePosition;//���ӣ���������������������
			numOfStep ++;//a new num that count the num of chess
			if (AIpos.x == 9) {
				ChessState = 0;
				if (IFend()) {
					goto finalJUD;
				}
				else { goto Hummanshow; }
			}
			allchess[AIpos.x][AIpos.y].state = 2;//���������ⲽ֮��ı䰴ť״̬
			ChessBoard[AIpos.x][AIpos.y] = Choosecolor;//ȷ������

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
			ReverseChess(JudjeWSTurn(-Choosecolor, AIpos.x, AIpos.y));//��ת�������
			//Sleep(2000);
			ChangeColor(ChessBoard);
			ChessState = 0;
			FlushBatchDraw();
			//ChessBoard[][] = -Choosecolor;
		}
	Hummanshow:	while (!ChessState) {//�ȴ��������ӣ�����ȡ����λ��
		WhoseTurn(&Infor, 1);
		drawButton(Infor);
		if (numOfStep == 20 || numOfStep ==21 || numOfStep == 22) { changeArr(); }
		if (numOfStep == 55 || numOfStep == 55 || numOfStep == 55) { changeArrAgain(); }
		Supervise Information = JWHHP(-Choosecolor);
		numOfStep++;//a new num that count the num of chess
		if (Information.state == 1) {//�����Ѿ�����,�Ϳ��Խ��з�ת��
			ChessState = 1;//��һ����������
		}//�ı���������
		else if (Information.state == 2) {//������������Ļ���
			ChessState = 1;//��AI����
			if (IFend()) {//���������
				//cout << "COMEIN" << endl;
				goto finalJUD;
			}
			break;
		}
		if (Information.state == 10) {//������Ȩ
			WhoseTurn(&Infor, 0); drawButton(Infor);
			Sleep(5000);
			return 0;
		}
		ChessBoard[Information.x][Information.y] = -Choosecolor;//��������ϵ�λ�����ȱ��������ˣ���ת��ɫΪChoosecolor������
		allchess[Information.x][Information.y].state = 2;
		ButtonSleep();
		ReverseChess(JudjeWSTurn(Choosecolor, Information.x, Information.y));//��ת�����˵���
		Sleep(100);
		ChangeColor(ChessBoard);
		//FlushBatchDraw();
	}//֮����ֵ�����������
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
	EndBatchDraw();//˫����
	return 0;
}
//�ҵ�Ŀ�ģ���3����ʼ�������Ĳ��������ü�֦�������̵�ÿ���������ɰ�ť�ṹ�壬û�µ�ʱ�������ť�����ã��¹�֮��ֻ�ı���ɫ
//������Ϸ���������ʾ��������ʱ����.
//�ѵ㣺�������飬���������ʱ���ı�Ԫ�أ�
//дһ���ӿڣ������ť���룬����ת�����λ�ô�����������λ�ô������������ڸı������ɫ�����Ҹ����Է����������λ��