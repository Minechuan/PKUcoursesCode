#ifndef CHOOSETHEME_H
#define CHOOSETHEME_H

#include"hoverbutton.h"
#include <QWidget>
#include <QCoreApplication>
#include"global.h"
#include <QApplication>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>
#include <QPropertyAnimation>
#include <QEnterEvent>
namespace Ui {
class ChooseTheme;
}

class ChooseTheme : public QWidget
{
    Q_OBJECT


public:

    explicit ChooseTheme(QWidget *parent = nullptr);
    ~ChooseTheme();

signals:
    void themeChanged();

private slots:
    void changebegin();

    void on_PKU_clicked();

private:
    Ui::ChooseTheme *ui;
};



#endif // CHOOSETHEME_H
