#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "calendarmc.h"
#include "clockhzj.h"
#include "dailymsy.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT


public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    //virtual void keyPressEvent(QKeyEvent * event);
    //void paintEvent(QPaintEvent *e);
    void setAW();

signals:
    void themeChanged();//在更改主题之后，传递更改完成的信号


private slots:
    void on_calBT_clicked();

    void on_clockBT_clicked();

    void on_pushButton_clicked();

    void on_TTBT_clicked();

    void on_readme_clicked();

    void on_style_clicked();

    void changetheme();

private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
