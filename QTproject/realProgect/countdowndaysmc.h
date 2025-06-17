#ifndef COUNTDOWNDAYSMC_H
#define COUNTDOWNDAYSMC_H

#include <QWidget>
#include <QDate>
#include<QDebug>
#include "calendarmc.h"


//这里会用到在calendarmc.h中创建的数据库
namespace Ui {
class CountDownDaysMC;
}

class CountDownDaysMC : public QWidget
{
    Q_OBJECT

public:
    void PrintForView();


public:
    explicit CountDownDaysMC(QWidget *parent = nullptr);
    ~CountDownDaysMC();
    void paintEvent(QPaintEvent *e);

signals:
    void themechanged();

public slots:
    void changetheme();//槽函数


private:
    Ui::CountDownDaysMC *ui;
};

#endif // COUNTDOWNDAYSMC_H
