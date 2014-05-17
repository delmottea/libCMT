#include <QCoreApplication>
#include <CMT.h>
#include <QDebug>

void num2str(char *str, int length, int num)
{
    for(int i = 0; i < length-1; i++)
    {
        str[length-i-2] = '0'+num%10;
        num /= 10;
    }
    str[length-1] = 0;
}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    char *path = "sequences/cokecan/img";
    char *ext = "png";
    int numLength = 5;
    char numString[numLength+1];
    char filename[255];
    int start = 0;
    int end = 291;

    qDebug() << CV_MAJOR_VERSION << CV_MINOR_VERSION;
    CMT cmt;
    cmt.estimateRotation = false;
    for(int i = start; i <= end; i++)
    {
        num2str(numString, numLength+1, i);
        sprintf(filename, "%s%s.%s", path, numString, ext);

        qDebug() << filename;

        cv::Mat img = cv::imread(filename);
        cv::Mat im_gray;
        cv::cvtColor(img, im_gray, CV_RGB2GRAY);

        if(i == start)
            cmt.initialise(im_gray, cv::Point2f(150,82), cv::Point2f(170,118));
        cmt.processFrame(im_gray);

        //cv::circle(img, cmt.activeKeypoints[0].first.pt, 5, cv::Scalar(255,255,255));
        for(int i = 0; i<cmt.trackedKeypoints.size(); i++)
        {
            cv::circle(img, cmt.trackedKeypoints[i].first.pt, 3, cv::Scalar(255,255,255));
        }
        cv::line(img, cmt.topLeft, cmt.topRight, cv::Scalar(255,255,255));
        cv::line(img, cmt.topRight, cmt.bottomRight, cv::Scalar(255,255,255));
        cv::line(img, cmt.bottomRight, cmt.bottomLeft, cv::Scalar(255,255,255));
        cv::line(img, cmt.bottomLeft, cmt.topLeft, cv::Scalar(255,255,255));

        //cv::rectangle(img, cv::Rect(cmt.boundingbox.x, cmt.boundingbox.y, cmt.boundingbox.width, cmt.boundingbox.height), cv::Scalar(255,255,255));

        imshow("frame", img);
        /*if(i == 1)
        {
            cv::waitKey(100000);
            return 0;
        }*/
        cv::waitKey(1);
    }
    return 0;//a.exec();
}
