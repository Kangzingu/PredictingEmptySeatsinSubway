#pragma once
// MFCApplication1_0605projectView.cpp : CMFCApplication1_0605projectView Ŭ������ ����
//
#include "stdafx.h"
#include "opencv2/opencv.hpp"

#include "opencv2/objdetect.hpp"//
#include "opencv2/videoio.hpp"//
#include "opencv2/highgui.hpp"//
#include "opencv2/imgproc.hpp"//

using namespace cv;
// SHARED_HANDLERS�� �̸� ����, ����� �׸� �� �˻� ���� ó���⸦ �����ϴ� ATL ������Ʈ���� ������ �� ������
// �ش� ������Ʈ�� ���� �ڵ带 �����ϵ��� �� �ݴϴ�.
#ifndef SHARED_HANDLERS
#include "MFCApplication1_0605project.h"
#endif

#include "MFCApplication1_0605projectDoc.h"
#include "MFCApplication1_0605projectView.h"

#include <vector>
#include "opencv/cv.h"
#include "opencv/highgui.h"
using namespace std;

#define M_PI 3.14159265358979323846
#define WIN_NAME "���ν�"
#define FACE_CLASSIFIER_PATH "C:/Users/kanrh/Downloads/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml"
#define FACE_SEARCH_SCALE 1.1
#define MERGE_DETECTED_GROUP_CNTS 3
#define FACE_FRAME_WIDTH 50
#define FACE_FRAME_HEIGHT 50
#define FACE_FRAME_THICKNESS 1
#define eyes_cascade_name = "C:/Users/kanrh/Downloads/opencv/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
//�ȱ� �ν� XML ���� ��� ����

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CMFCApplication1_0605projectView

IMPLEMENT_DYNCREATE(CMFCApplication1_0605projectView, CView)

BEGIN_MESSAGE_MAP(CMFCApplication1_0605projectView, CView)
	// ǥ�� �μ� ����Դϴ�.
	ON_COMMAND(ID_FILE_PRINT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, &CView::OnFilePrintPreview)
	ON_COMMAND(ID_LOADAVI, &CMFCApplication1_0605projectView::OnLoadavi)
	ON_COMMAND(ID_BACKGROUNDDETECTION, &CMFCApplication1_0605projectView::OnBackgrounddetection)
	ON_COMMAND(ID_HANDDETECTION, &CMFCApplication1_0605projectView::OnHanddetection)
	ON_COMMAND(ID_FACEDETECTION, &CMFCApplication1_0605projectView::OnFacedetection)
	ON_COMMAND(ID_EDGEDETECTION, &CMFCApplication1_0605projectView::OnEdgedetection)
	ON_COMMAND(ID_LABELING, &CMFCApplication1_0605projectView::OnLabeling)
	ON_COMMAND(ID_HANDLABELING, &CMFCApplication1_0605projectView::OnHandlabeling)
	ON_COMMAND(ID_BACKGROUNDSUBWAY, &CMFCApplication1_0605projectView::OnBackgroundsubway)
END_MESSAGE_MAP()
//Mat getHandMask1(const Mat& image, int minCr = 128, int maxCr = 170, int minCb = 73, int maxCb = 158);
// CMFCApplication1_0605projectView ����/�Ҹ�
Point getHandCenter(const Mat& mask, double& radius);
Mat DrawLabelingImage(Mat image);
Mat onSobel(Mat rgbImg);
Mat onGrayMedian(Mat rgbImg);
Mat onMedian(Mat rgbImg);
Mat EdgeImprove(Mat image);
void DrawLabelingImageEye(Mat img_gray, int(*connectedLabels)[4], bool& connected);
Mat DrawLabelingImageHand(Mat img_gray, int*handLabels, bool& substracted);

Mat hand3ss;
bool handFlag = false;
//Mat input3ss;
void On3ss(Mat frame);
RGBQUAD** aviBuffer1;
RGBQUAD** aviBuffer2;
int aviHeight;
int aviWidth;
int cnt = 0;
int allowCnt = 0;
int frameCnt = 0;

class ArrValue {
public:
	int x_dif = 0;
	int y_dif = 0;
};
ArrValue On3SS(int x, int y, ArrValue av, int number, int n);
double calculate_subtraction(int x, int y, int x_dif, int y_dif, int n);

CMFCApplication1_0605projectView::CMFCApplication1_0605projectView()


{
	// TODO: ���⿡ ���� �ڵ带 �߰��մϴ�.

}

CMFCApplication1_0605projectView::~CMFCApplication1_0605projectView()
{
}

BOOL CMFCApplication1_0605projectView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: CREATESTRUCT cs�� �����Ͽ� ���⿡��
	//  Window Ŭ���� �Ǵ� ��Ÿ���� �����մϴ�.

	return CView::PreCreateWindow(cs);
}

// CMFCApplication1_0605projectView �׸���

void CMFCApplication1_0605projectView::OnDraw(CDC* /*pDC*/)
{
	CMFCApplication1_0605projectDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	// TODO: ���⿡ ���� �����Ϳ� ���� �׸��� �ڵ带 �߰��մϴ�.
}


// CMFCApplication1_0605projectView �μ�

BOOL CMFCApplication1_0605projectView::OnPreparePrinting(CPrintInfo* pInfo)
{
	// �⺻���� �غ�
	return DoPreparePrinting(pInfo);
}

void CMFCApplication1_0605projectView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: �μ��ϱ� ���� �߰� �ʱ�ȭ �۾��� �߰��մϴ�.
}

void CMFCApplication1_0605projectView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: �μ� �� ���� �۾��� �߰��մϴ�.
}


// CMFCApplication1_0605projectView ����

#ifdef _DEBUG
void CMFCApplication1_0605projectView::AssertValid() const
{
	CView::AssertValid();
}

void CMFCApplication1_0605projectView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CMFCApplication1_0605projectDoc* CMFCApplication1_0605projectView::GetDocument() const // ����׵��� ���� ������ �ζ������� �����˴ϴ�.
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CMFCApplication1_0605projectDoc)));
	return (CMFCApplication1_0605projectDoc*)m_pDocument;
}
#endif //_DEBUG


// CMFCApplication1_0605projectView �޽��� ó����


void CMFCApplication1_0605projectView::OnLoadavi()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CFileDialog dlg(TRUE, ".avi", NULL, NULL, "AVI File (*.avi)|*.avi||");
	if (IDOK != dlg.DoModal())
		return;
	CString cfilename = dlg.GetPathName();
	CT2CA strAtl(cfilename);
	String filename(strAtl);

	cv::VideoCapture Capture;
	Capture.open(filename);
	if (!Capture.isOpened())
		AfxMessageBox("Error Video");

	for (;;) {
		Mat frame;
		Capture >> frame;
		if (frame.data == nullptr)
			break;
		imshow("video", frame);
		if (waitKey(30) >= 0)
			break;
	}
	//AfxMessageBox("Completed");
}
void CMFCApplication1_0605projectView::OnBackgrounddetection()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CFileDialog dlg(TRUE, ".avi", NULL, NULL, "AVI File (*.avi)|*.avi||");
	if (IDOK != dlg.DoModal())
		return;
	// Init background substractor
	CString cfilename = dlg.GetPathName();
	CT2CA strAtl(cfilename);
	String filename(strAtl);

	cv::VideoCapture Capture;
	Capture.open(filename);
	if (!Capture.isOpened())
		AfxMessageBox("Error Video");
	Ptr<BackgroundSubtractor> bg_model = createBackgroundSubtractorMOG2().dynamicCast<BackgroundSubtractor>();
	//MOG2 �Լ��� �̿��� ��� ����
	Mat foregroundMask, backgroundImage, foregroundImg;
	//////////////////////////VideoCapture cap(0);
	int sumWhite = 0; int sumBlack = 0;
	CString result;
	for (;;) {
		Mat img;
		Capture >> img;
		//������ �ҷ�����
		if (img.data == nullptr)
			break;
		resize(img, img, Size(480, 320));
		if (foregroundMask.empty()) {
			foregroundMask.create(img.size(), img.type());
		}
		bg_model->apply(img, foregroundMask, true ? -1 : 0);
		GaussianBlur(foregroundMask, foregroundMask, Size(11, 11), 3.5, 3.5);
		//��ó���� ���� ����
		threshold(foregroundMask, foregroundMask, 10, 255, THRESH_BINARY);
		foregroundImg = Scalar::all(0);
		img.copyTo(foregroundImg, foregroundMask);
		bg_model->getBackgroundImage(backgroundImage);
		//����̹����� ������
		imshow("foreground mask", foregroundMask);
		//�츮�� ���ϴ� ��� ��ü����
		imshow("foreground image", foregroundImg);
		int key6 = waitKey(40);
		if (!backgroundImage.empty()) {
			imshow("background image", backgroundImage);
			int key5 = waitKey(40);
		}
		imshow("original image", img);
		for (int i = 0; i < 320; i++) {
			for (int j = 0; j < 480; j++) {
				if (foregroundMask.ptr<BYTE>(i, j)[0] == 0)
					sumBlack++;
				else
					sumWhite++;
			}
		}
		CString paste;
		float sum = (float)sumWhite / (sumBlack + sumWhite);
		//����� ������ ����
		paste.Format("(%f)\n", sum);
		result += paste;
		//��� �غ�
		CStdioFile file;
		file.Open(_T("FILE.TXT"), CFile::modeCreate | CFile::modeWrite);
		//������������ ����, Ȥ�ó� ���ϸ� ��ã���� �ش� ���� ������ ���� �����
		file.WriteString(result);
		file.Close();
	}
}
void CMFCApplication1_0605projectView::OnHanddetection()
{
	Mat resultHandImg, tmpImg, originalHandImg;
	VideoCapture video(0);
	namedWindow("result_hand_image", CV_WINDOW_AUTOSIZE);
	namedWindow("original_hand_image", CV_WINDOW_AUTOSIZE);
	namedWindow("result_face_image", CV_WINDOW_AUTOSIZE);
	//���â
	Mat resultHandCenterImg;
	CascadeClassifier face_classifier;
	face_classifier.load(FACE_CLASSIFIER_PATH);
	//�������
	Mat resultFaceImg;
	vector<Rect> faces;
	//���� ����
	while (true) {//�ݺ�
		video >> tmpImg;//�׳� ���� �ӽ������
		video >> resultHandCenterImg;//�չٴ� �ν� ��� �����
		bool isFrameValid = true;
		try {
			// �� ķ �������� ���� ũ�� ����
			video >> tmpImg;

			// ���� ũ���� 1/2�� ��� �Ҽ���¡�� ���� ���� �����ž� (�ֳĸ� �������� ũ�Ⱑ Ŭ ��� ����ð��� ����)
			resize(tmpImg, resultFaceImg, cv::Size(tmpImg.cols / 1, tmpImg.rows / 1), 0, 0, CV_INTER_NN);
		}
		catch (cv::Exception& e) {
			// ���� ���
			std::cerr << "������ ��ҿ� �����߱⿡, �ش� �������� �����մϴ�." << e.err << std::endl;
			isFrameValid = false;
		}
		if (isFrameValid) {
			try {
				Mat grayframe;
				//�׷��� ������ �̹����� ����
				cvtColor(resultFaceImg, grayframe, CV_BGR2GRAY);
				// �������� �׷��� ������ �� ���������� ó��
				equalizeHist(grayframe, grayframe);


				// ���ν� ���ø��� �̿��Ͽ� ���ν�
				face_classifier.detectMultiScale(
					grayframe, faces,
					FACE_SEARCH_SCALE,
					MERGE_DETECTED_GROUP_CNTS,
					CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE,
					Size(FACE_FRAME_WIDTH, FACE_FRAME_HEIGHT)
				);

				for (int i = 0; i < faces.size(); i++) {

					// ���ν� �簢�� Ʋ ���
					Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
					Point tr(faces[i].x, faces[i].y);
					rectangle(resultFaceImg, lb, tr, Scalar(0, 0, 255), FACE_FRAME_THICKNESS, 4, 0);

				}
				// �����쿡 ��� ���
				imshow("original image", tmpImg);
				imshow("result_face_image", resultFaceImg);
			}
			catch (cv::Exception& e) {
				cerr << "���ν� ó���� �����߱⿡, �ش� �������� �����մϴ�." << e.err << endl;
			}

		}
		//
		try {
			cvtColor(tmpImg, resultHandImg, CV_BGR2YCrCb);
			//�Ǻ� �� ���� ����
			inRange(resultHandImg, Scalar(0, 133, 77), Scalar(255, 173, 127), resultHandImg);

			//8bit ����ä��?
			originalHandImg = (resultHandImg.size(), CV_8UC3, Scalar(0));

			add(tmpImg, Scalar(0), originalHandImg, resultHandImg);

			imshow("result_hand_image", resultHandImg);
			imshow("original_hand_image", originalHandImg);
			erode(resultHandImg, resultHandImg, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2);//
			double radius;
			for (int i = 0; i < faces.size(); i++) {
				//�� �ν� �� �� ����
				Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
				Point tr(faces[i].x, faces[i].y);
				for (int j = 0; j < faces[i].height; j++) {
					for (int k = 0; k < faces[i].width; k++) {
						resultHandImg.ptr<BYTE>(tr.y + k, tr.x + j)[0] = 0;
						resultHandImg.ptr<BYTE>(tr.y + k, tr.x + j)[1] = 0;
						resultHandImg.ptr<BYTE>(tr.y + k, tr.x + j)[2] = 0;
					}
				}
			}
			//�չٴ� �߽� �� ��ȯ
			Point center = getHandCenter(resultHandImg, radius);
			//������ �� ǥ��
			circle(resultHandCenterImg, center, 2, Scalar(0, 255, 0), -1);
			circle(resultHandCenterImg, center, (int)(radius + 0.5), Scalar(255, 0, 0), 2);
			imshow("handcenter_image", resultHandCenterImg);
		}
		catch (Exception& e) {
			cerr << "ó���� �����߱⿡, �ش� �������� �����մϴ�." << e.err << endl;
		}

		if (waitKey(27) == 27)
			break;
	}
	//�޸� ����
	video.release();
	tmpImg.release();
	resultHandImg.release();
	originalHandImg.release();
	destroyAllWindows();
}
Mat onMedian(Mat rgbImg) {
	float** hueBuffer;
	float** satuBuffer;
	float** intenBuffer;
	float** mediBuffer;
	int imgHeight = rgbImg.rows;
	int imgWidth = rgbImg.cols;
	hueBuffer = new float*[imgHeight];//���Ǽ���
	satuBuffer = new float*[imgHeight];//���Ǽ���
	intenBuffer = new float*[imgHeight];//���Ǽ���
	mediBuffer = new float*[imgHeight];//���Ǽ���
	for (int i = 0; i < imgHeight; i++) {
		hueBuffer[i] = new float[imgWidth];
		satuBuffer[i] = new float[imgWidth];
		intenBuffer[i] = new float[imgWidth];
		mediBuffer[i] = new float[imgWidth];
	}
	for (int i = 0; i < imgHeight; i++) {
		for (int j = 0; j < imgWidth; j++) {
			float r = rgbImg.ptr<BYTE>(i, j)[2] / 255.0;
			float g = rgbImg.ptr<BYTE>(i, j)[1] / 255.0;
			float b = rgbImg.ptr<BYTE>(i, j)[0] / 255.0;

			intenBuffer[i][j] = (1.0 / 3.0)*(r + g + b);
			float minRGB;
			if (r > g) {
				minRGB = g;
				if (minRGB > b)
					minRGB = b;
			}
			else {
				minRGB = r;
				if (minRGB > b)
					minRGB = b;
			}
			satuBuffer[i][j] = 1.0 - (3.0 / (float)(r + g + b))*(minRGB);
			hueBuffer[i][j] = acos(((1.0 / 2.0)*((r - g) + (r - b))) / (sqrt((r - g)*(r - g) + (r - b)*(g - b))));
			hueBuffer[i][j] *= 57.29577951;
			if (b > g)
				hueBuffer[i][j] = 360 - hueBuffer[i][j];
			if (hueBuffer[i][j] < 0)
				hueBuffer[i][j] += 360;
			else if (hueBuffer[i][j] > 360)
				hueBuffer[i][j] -= 360;
		}
	}
	//
	float table[25] = { 0 };
	for (int i = 2; i < imgHeight - 2; i++) {
		for (int j = 2; j < imgWidth - 2; j++) {
			table[0] = intenBuffer[i - 2][j - 2];
			table[1] = intenBuffer[i - 2][j - 1];
			table[2] = intenBuffer[i - 2][j];
			table[3] = intenBuffer[i - 2][j + 1];
			table[4] = intenBuffer[i - 2][j + 2];
			table[5] = intenBuffer[i - 1][j - 2];
			table[6] = intenBuffer[i - 1][j - 1];
			table[7] = intenBuffer[i - 1][j];
			table[8] = intenBuffer[i - 1][j + 1];
			table[9] = intenBuffer[i - 1][j + 2];
			table[10] = intenBuffer[i][j - 2];
			table[11] = intenBuffer[i][j - 1];
			table[12] = intenBuffer[i][j];
			table[13] = intenBuffer[i][j + 1];
			table[14] = intenBuffer[i][j + 2];
			table[15] = intenBuffer[i + 1][j - 2];
			table[16] = intenBuffer[i + 1][j - 1];
			table[17] = intenBuffer[i + 1][j];
			table[18] = intenBuffer[i + 1][j + 1];
			table[19] = intenBuffer[i + 1][j + 2];
			table[20] = intenBuffer[i + 2][j - 2];
			table[21] = intenBuffer[i + 2][j - 1];
			table[22] = intenBuffer[i + 2][j];
			table[23] = intenBuffer[i + 2][j + 1];
			table[24] = intenBuffer[i + 2][j + 2];
			for (int k = 0; k < 25; k++) {
				for (int l = k; l < 25; l++) {
					if (table[k] > table[l]) {
						float trash = table[k];
						table[k] = table[l];
						table[l] = trash;
					}
				}
			}
			mediBuffer[i][j] = table[12];
		}
	}
	//
	for (int i = 0; i < imgHeight; i++) {
		for (int j = 0; j < imgWidth; j++) {
			float H = hueBuffer[i][j];
			float S = satuBuffer[i][j];
			float I = mediBuffer[i][j];
			float r = 0;
			float g = 0;
			float b = 0;
			if (0 <= H && H < 120) {
				float ceta = H*0.017453293;
				float ceta2 = (60.0 - H)*0.017453293;
				b = (1.0 / 3.0)*(float)(1 - S);
				r = (1.0 / 3.0)*(float)(1 + ((float)(S*cos(ceta)) / (float)(cos(ceta2))));
				g = 1 - (r + b);
			}
			else if (120 <= H&&H < 240) {
				H = H - 120;
				float ceta = H*0.017453293;
				float ceta2 = (60.0 - H)*0.017453293;
				g = (1.0 / 3.0)*(float)(1 + ((float)(S*cos(ceta)) / (float)(cos(ceta2))));
				r = (1.0 / 3.0)*(float)(1 - S);
				b = 1 - (r + g);
			}
			else if (240 <= H&&H <= 360) {
				H = H - 240;
				float ceta = H*0.017453293;
				float ceta2 = (60.0 - H)*0.017453293;
				b = (1.0 / 3.0)*(float)(1 + (float)((float)(S*cos(ceta)) / (float)(cos(ceta2))));
				g = (1.0 / 3.0)*(float)(1 - S);
				r = 1 - (g + b);
			}
			rgbImg.ptr<BYTE>(i, j)[2] = 3.0*r*I*255.0;
			rgbImg.ptr<BYTE>(i, j)[1] = 3.0*g*I*255.0;
			rgbImg.ptr<BYTE>(i, j)[0] = 3.0*b*I*255.0;
		}
	}
	for (int i = 0; i < imgHeight; i++) {
		delete[] hueBuffer[i];
		delete[] satuBuffer[i];
		delete[] intenBuffer[i];
		delete[] mediBuffer[i];
	}
	delete[] hueBuffer;
	delete[] satuBuffer;
	delete[] intenBuffer;
	delete[] mediBuffer;
	return rgbImg;
}
Point getHandCenter(const Mat& mask, double& radius) {
	//�Ÿ� ��ȯ ����� ������ ����
	Mat dst;
	distanceTransform(mask, dst, CV_DIST_L2, 5);
	//����� CV_32SC1 Ÿ��
	//�Ÿ� ��ȯ ��Ŀ��� ��(�Ÿ�)�� ���� ū �ȼ��� ��ǥ��, ���� ���´�.
	int maxIdx[2];//��ǥ ���� ���� �迭(��, �� ������ �����)
	minMaxIdx(dst, NULL, &radius, NULL, maxIdx, mask);//�ּҰ��� ��� X
	return Point(maxIdx[1], maxIdx[0]);

}
void CMFCApplication1_0605projectView::OnFacedetection()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	//�޴� ����� ���� �� ȣ��Ǵ� �Լ�

	// �� ķ ����
	VideoCapture capture(0);

	// �� ķ�� �������� ���� ��� ���� ��� �� ����
	if (!capture.isOpened()) {
		cerr << "ERROR : �� ķ ����̽��� ã�� �� ����" << std::endl;
		//return;//�����մϴ�.
	}

	// ������ ����
	namedWindow(WIN_NAME, 1);

	// ���ν� ���ø� ����
	CascadeClassifier face_classifier;
	face_classifier.load(FACE_CLASSIFIER_PATH);

	Mat frameOriginalMat;
	Mat frame;
	vector<Rect> faces;
	while (true) {

		bool isFrameValid = true;
		try {
			// �� ķ �������� ���� ũ�� ����
			capture >> frameOriginalMat;

			// ���� ũ���� 1/2�� ��� (�ֳĸ� �������� ũ�Ⱑ Ŭ ��� ����ð��� ����)
			resize(frameOriginalMat, frame, cv::Size(frameOriginalMat.cols / 2, frameOriginalMat.rows / 2), 0, 0, CV_INTER_NN);
		}
		catch (cv::Exception& e) {
			// ���� ���
			std::cerr << "������ ��ҿ� �����߱⿡, �ش� �������� �����մϴ�." << e.err << std::endl;
			isFrameValid = false;
		}

		// ������ ũ�� ��ҿ� ������ ��� ���ν�
		if (isFrameValid) {
			try {
				// �������� �׷��� ������ �� ���������� ó��
				Mat grayframe;
				cvtColor(frame, grayframe, CV_BGR2GRAY);
				equalizeHist(grayframe, grayframe);

				// ���ν� ���ø��� �̿��Ͽ� ���ν�
				face_classifier.detectMultiScale(
					grayframe, faces,
					FACE_SEARCH_SCALE,
					MERGE_DETECTED_GROUP_CNTS,
					CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE,
					Size(FACE_FRAME_WIDTH, FACE_FRAME_HEIGHT)
				);
				for (int i = 0; i < faces.size(); i++) {
					// ���ν� �簢�� Ʋ ���
					Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
					Point tr(faces[i].x, faces[i].y);
					rectangle(frame, lb, tr, Scalar(0, 0, 255), FACE_FRAME_THICKNESS, 4, 0);
				}

				// �����쿡 ��� ���
				cv::imshow(WIN_NAME, frame);
			}
			catch (cv::Exception& e) {
				cerr << "���ν� ó���� �����߱⿡, �ش� �������� �����մϴ�." << e.err << endl;
			}
		}
		int keyCode = cv::waitKey(30);

		// esc Ű�� ������ ������ ĸ�� ����
		if (keyCode == 27) {
			break;
		}
	}


	return;//�����մϴ�

}
void CMFCApplication1_0605projectView::OnEdgedetection()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	Mat resultHandImg, tmpImg, originalHandImg;
	VideoCapture video(0);
	namedWindow("result_hand_image", CV_WINDOW_AUTOSIZE);
	namedWindow("edge_hand_result_image", CV_WINDOW_AUTOSIZE);
	namedWindow("result_face_image", CV_WINDOW_AUTOSIZE);
	namedWindow("edge_detect_image", CV_WINDOW_AUTOSIZE);
	namedWindow("median_edge_detect_image", CV_WINDOW_AUTOSIZE);
	//���â
	Mat resultHandCenterImg;
	Mat medianImg, edgeImg;
	//medianImg�� �޵�� ���� �� �����𶙰��������
	//endeImg�� �����𶙰��������
	//���� ����
	while (true) {//�ݺ�
		video >> tmpImg;//�׳� ���� �ӽ������
		video >> resultHandCenterImg;//�չٴ� �ν� ��� �����
		try {
			cvtColor(tmpImg, resultHandImg, CV_BGR2YCrCb);
			//�Ǻ� �� ���� ����
			inRange(resultHandImg, Scalar(0, 133, 77), Scalar(255, 173, 127), resultHandImg);

			//8bit ����ä��?
			originalHandImg = (resultHandImg.size(), CV_8UC3, Scalar(0));
			edgeImg = onSobel(~tmpImg);

			tmpImg = onMedian(tmpImg);
			medianImg = onSobel(~tmpImg);
			add(edgeImg, Scalar(0), originalHandImg, resultHandImg);
			//tmpImg=onRGBToHSI(tmpImg);
			//���� ��
			cvtColor(medianImg, medianImg, CV_BGR2GRAY);
			cvtColor(edgeImg, edgeImg, CV_BGR2GRAY);
			//�׷��� �����Ϸ� ����
			imshow("median_edge_detect_image", medianImg);//
			imshow("edge_detect_image", edgeImg);
			imshow("result_hand_image", resultHandImg);
			imshow("edge_hand_result_image", originalHandImg);//�� �� ������ �����Ŷ� ���������Ѱ� ��ģ��
			erode(resultHandImg, resultHandImg, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2);//
			double radius;
			//�չٴ� �߽� �� ��ȯ
			Point center = getHandCenter(resultHandImg, radius);
			//������ �� ǥ��
			circle(resultHandCenterImg, center, 2, Scalar(0, 255, 0), -1);
			circle(resultHandCenterImg, center, (int)(radius + 0.5), Scalar(255, 0, 0), 2);
			imshow("handcenter_image", resultHandCenterImg);
		}
		catch (Exception& e) {
			cerr << "ó���� �����߱⿡, �ش� �������� �����մϴ�." << e.err << endl;
		}

		if (waitKey(27) == 27)
			break;
	}
	//�޸� ����
	video.release();
	tmpImg.release();
	resultHandImg.release();
	originalHandImg.release();
	destroyAllWindows();
}

void CMFCApplication1_0605projectView::OnLabeling(){
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	VideoCapture video(0);
	Mat tmpImg;//������, �Ŀ� ����� �������� ���� ���ϰ���
	Mat textureImg;//�ؽ���
	Mat textureMask;//�ؽ��� ����ũ
	Mat edgeImg;//����
	Mat result;//���
	while (true) {//�ݺ�
		video >> tmpImg;//������ �޾ƿ´�
		resize(tmpImg, tmpImg, Size(480, 320));
		try {
			namedWindow("original", WINDOW_AUTOSIZE);
			imshow("original", tmpImg);
			//ĸ���س� ���� ���
			cvtColor(tmpImg, textureImg, CV_BGR2YCrCb);
			//YCrCb �����̽��� ����
			inRange(textureImg, Scalar(0, 133, 77), Scalar(255, 173, 127), textureImg);
			//������ �Ǻλ� ������ ��, �� �ν� ����� grayscale �̹�����
			threshold(textureImg, textureMask, 127, 255, THRESH_BINARY);
			//����� ���� grayscale �̹����� threshold�ؼ� �̾Ƴ�, �Ŀ� ����ũ�� ����Ұ���
			//////////////////////////////////////////////////////////////////////////////////////////////
			//add(tmpImg, Scalar(0), textureImg, textureMask);
			//������ ���� ����� ����ũ�� �̿��� �Ǻ� �̿ܿ� �κ��� ���������� ĥ�ع���
			//add(~textureMask, Scalar(0), textureImg, ~textureMask);
			//�״㿡 �Ǻθ� ������� ĥ�� �ᱹ�� �Ǻ�=���, �׿�=������ ���� ����///////////////////////// �ٵ� �� ������ �ʿ䰡 ������;
			//////////////////////////////////////////////////////////////////////////////////////////////
			result = ~onGrayMedian(textureMask);
			//�׳� ����ũ �̾Ƴ��� ��ü�� �Ǻζ� ������ �����ִ� ���̳ʸ� �̹����� 
			//�̷��� �̰� �Ѱܼ� median �Լ��� �̿��� Ƣ�°� �����ش�

			edgeImg = onSobel(tmpImg);
			//ó������ ���ư� ���������� ���������ش�
			cvtColor(edgeImg, edgeImg, CV_BGR2GRAY);
			//�� ���� �̾Ƴ��� �׷��� �����Ϸ� �����Ѵ�
			threshold(edgeImg, edgeImg, 127, 255, THRESH_BINARY);
			//������Ȧ�� �ؼ� ����=���, �׿�=������ ���� ����
			edgeImg = EdgeImprove(~edgeImg);
			//������ �ִ� �ȼ��� ũ�� Ű���� ���� �� �̾������� ���ش�

			add(~edgeImg, Scalar(0), result, ~edgeImg);
			//���� ������ ��+�� ������ ���ļ� ��, �� ���еǵ��� �ؼ�
			DrawLabelingImage(~result);
			//�󺧵� �̾Ƴ���, �ڽ�ġ�� �׷��� �� �Լ� �ȿ��� �˾Ƽ� ���� ��
			/*namedWindow("sibal13", WINDOW_AUTOSIZE);
			imshow("sibal13", tmpImg);*/
		}
		catch (Exception& e) {
			cerr << "ó���� �����߱⿡, �ش� �������� �����մϴ�." << e.err << endl;
		}

		if (waitKey(27) == 27)
			break;
	}
	//�޸� ����
	video.release();
	tmpImg.release();
	textureImg.release();
	textureMask.release();
	edgeImg.release();
	result.release();
	destroyAllWindows();
}
Mat onGrayMedian(Mat rgbImg) {
	int imgHeight = rgbImg.rows;
	int imgWidth = rgbImg.cols;
	float table[9] = { 0 };
	for (int i = 1; i < imgHeight - 1; i++) {
		for (int j = 1; j < imgWidth - 1; j++) {
			for (int k = -1; k < 2; k++) {
				for (int l = -1; l < 2; l++) {
					table[(k + 1) * 3 + (l + 1)] = rgbImg.at<uchar>(i + k, j + l);
				}
			}
			for (int k = 0; k < 9; k++) {
				for (int l = k; l < 9; l++) {
					if (table[k] > table[l]) {
						float trash = table[k];
						table[k] = table[l];
						table[l] = trash;
					}
				}
			}
			rgbImg.at<uchar>(i, j) = table[4];
		}
	}
	return rgbImg;
}
Mat EdgeImprove(Mat image) {//�ȼ� Ű���(���� ������ �߹��̰� �Ϸ���)
	int imgHeight = image.rows;
	int imgWidth = image.cols;
	int flag = 5;
	for (int i = flag; i < imgHeight; i++) {
		for (int j = flag; j < imgWidth; j++) {
			if (image.at<uchar>(i, j) < 128) {
				//���� �� �ڽ��� ������ 24ĭ�� �ڽ��� ������ ä��
				for (int k = -flag; k < 1; k++) {
					for (int l = -flag; l < 1; l++) {
						image.at<uchar>(i + k, j + l) = image.at<uchar>(i, j);
					}
				}
			}
		}
	}
	return image;
}
Mat DrawLabelingImageHand(Mat img_gray, int* handLabels, bool& substracted, int* resultHandLabels)
{
	//�ƿ�ǲ ���� 2���迭 [2][4]

	Mat img_color;
	Mat img_binary;
	threshold(img_gray, img_binary, 127, 255, THRESH_BINARY);
	cvtColor(img_gray, img_color, COLOR_GRAY2BGR);


	Mat img_labels, stats, centroids;
	int numOfLables = connectedComponentsWithStats(img_binary, img_labels,
		stats, centroids, 4, CV_32S);
	int sumOfBigLables = -1;//�ʱⰪ -1
	int **BigLabels = new int*[numOfLables];
	//������ �Ÿ��� ū�Ÿ� ���� �迭 �ϴ� ������� �𸣴� ������ ������ ������ ������
	//int connectedLabels[2][4];
	//ū���߿� ������?(y�� �����)�ֵ� ����
	//[4]�� ���� ���� x, y, �󺧿� �׸�ڽ� ������ height, width ������
	for (int i = 0; i < numOfLables; i++)
		BigLabels[i] = new int[4];
	//[4]�� ���� ���� x, y, �󺧿� �׸�ڽ� ������ height, width ������

	//�󺧸��� �̹����� Ư�� ���� �÷��� ǥ�����ֱ� �� ����� �Ⱦ���?��
	/*for (int y = 0; y<img_labels.rows; ++y) {

	int *label = img_labels.ptr<int>(y);
	Vec3b* pixel = img_color.ptr<Vec3b>(y);


	for (int x = 0; x < img_labels.cols; ++x) {


	if (label[x] == 3) {
	pixel[x][2] = 0;
	pixel[x][1] = 255;
	pixel[x][0] = 0;
	}
	}
	}
	*/

	//�󺧸� �� �̹����� ���� ���簢������ �ѷ��α� 
	for (int j = 1; j < numOfLables; j++) {
		int area = stats.at<int>(j, CC_STAT_AREA);
		int left = stats.at<int>(j, CC_STAT_LEFT);
		int top = stats.at<int>(j, CC_STAT_TOP);
		int width = stats.at<int>(j, CC_STAT_WIDTH);
		int height = stats.at<int>(j, CC_STAT_HEIGHT);
		if (width < 20 || height < 20) {
			continue;//����ũ�� ���ϸ� ����
		}
		/*CString str;
		str.Format("%d", left, top, width);
		AfxMessageBox(str);
*/
		if (sumOfBigLables == -1) {//ó�� ���³�(�Ӹ�) ����
			sumOfBigLables++;
			continue;
		}


		int x = centroids.at<double>(j, 0); //�߽���ǥ
		int y = centroids.at<double>(j, 1);

		BigLabels[sumOfBigLables][0] = left;
		BigLabels[sumOfBigLables][1] = top;
		BigLabels[sumOfBigLables][2] = width;
		BigLabels[sumOfBigLables][3] = height;
		sumOfBigLables++;//ī��Ʈ�� ����

		//circle(img_color, Point(x, y), 10, Scalar(255, 0, 0), 1);

		//rectangle(img_color, Point(left, top), Point(left + width, top + height),
		//	Scalar(0, 0, 255), 1);

		//putText(img_color, to_string(sumOfBigLables), Point(left + 20, top + 20), FONT_HERSHEY_SIMPLEX,
		//	1, Scalar(255, 0, 0), 2);
	}
	if (sumOfBigLables > 1)
		sumOfBigLables = 1;
	//2�� �̻��� �ʿ������ �ִ� 2���� �������� �ϻ꼭�� ��ȭ�� 2026 �츮��^^

	for (int i = 0; i < sumOfBigLables; i++) {
		handLabels[0] = BigLabels[i][0];
		handLabels[1] = BigLabels[i][1];
		handLabels[2] = BigLabels[i][2];
		handLabels[3] = BigLabels[i][3];
		//rectangle(img_color, Point(handLabels[0], handLabels[1]), Point(handLabels[0]+ handLabels[2], handLabels[1]+ handLabels[3]),
		//	Scalar(0, 0, 255), 1);
	}
	namedWindow("Hand Labeling Image", WINDOW_AUTOSIZE);             // Create a window for display
	imshow("Hand Labeling Image", img_color);
	frameCnt++;// Show our image inside it
	if (frameCnt > 4) {
		if (handFlag == false && sumOfBigLables != 0)
		{

			Mat handLabeled;
			Rect rect(Point(handLabels[0], handLabels[1]), Point(handLabels[0] + handLabels[2], handLabels[1] + handLabels[3]));
			hand3ss = img_color(rect);//Rect(handLabels[0], handLabels[1], handLabels[0] + handLabels[2], handLabels[1] + handLabels[3]));
			resultHandLabels[0] = handLabels[0];
			resultHandLabels[1] = handLabels[1];
			resultHandLabels[2] = handLabels[2];
			resultHandLabels[3] = handLabels[3];

			handFlag = true;// Show our image inside it
	//		namedWindow("3", WINDOW_AUTOSIZE);             // Create a window for display
	//		imshow("3", hand3ss);                        // Show our image inside it

		}

	}
	//Rect newRect(Point(handLabels[0], handLabels[1]), Point(handLabels[0] + handLabels[2], handLabels[1] + handLabels[3]));
	//Mat newHand = img_color(newRect);
	//���ο� �ڵ�
	for (int i = 0; i < numOfLables; i++)
		delete[] BigLabels[i];
	delete[] BigLabels;
	//�޸� ����
	return img_color;//���ο� ȭ�� ��ȯ
}

void DrawLabelingImageEye(Mat img_gray, int(*connectedLabels)[4], bool& connected)
{
	//�ƿ�ǲ ���� 2���迭 [2][4]
	Mat img_color;
	Mat img_binary;
	threshold(img_gray, img_binary, 127, 255, THRESH_BINARY);
	cvtColor(img_gray, img_color, COLOR_GRAY2BGR);


	Mat img_labels, stats, centroids;
	int numOfLables = connectedComponentsWithStats(img_binary, img_labels,
		stats, centroids, 4, CV_32S);
	int sumOfBigLables = -1;//�ʱⰪ -1
	int **BigLabels = new int*[numOfLables];
	//������ �Ÿ��� ū�Ÿ� ���� �迭 �ϴ� ������� �𸣴� ������ ������ ������ ������
	//int connectedLabels[2][4];
	//ū���߿� ������?(y�� �����)�ֵ� ����
	//[4]�� ���� ���� x, y, �󺧿� �׸�ڽ� ������ height, width ������
	for (int i = 0; i < numOfLables; i++)
		BigLabels[i] = new int[4];
	//[4]�� ���� ���� x, y, �󺧿� �׸�ڽ� ������ height, width ������

	//�󺧸��� �̹����� Ư�� ���� �÷��� ǥ�����ֱ� �� ����� �Ⱦ���?��
	/*for (int y = 0; y<img_labels.rows; ++y) {

	int *label = img_labels.ptr<int>(y);
	Vec3b* pixel = img_color.ptr<Vec3b>(y);


	for (int x = 0; x < img_labels.cols; ++x) {


	if (label[x] == 3) {
	pixel[x][2] = 0;
	pixel[x][1] = 255;
	pixel[x][0] = 0;
	}
	}
	}
	*/

	//�󺧸� �� �̹����� ���� ���簢������ �ѷ��α� 
	for (int j = 1; j < numOfLables; j++) {
		int area = stats.at<int>(j, CC_STAT_AREA);
		int left = stats.at<int>(j, CC_STAT_LEFT);
		int top = stats.at<int>(j, CC_STAT_TOP);
		int width = stats.at<int>(j, CC_STAT_WIDTH);
		int height = stats.at<int>(j, CC_STAT_HEIGHT);
		if (width < 6 || height < 6) {
			continue;//����ũ�� ���ϸ� ����
		}

		if (sumOfBigLables == -1) {
			sumOfBigLables++;
			continue;
		}


		int x = centroids.at<double>(j, 0); //�߽���ǥ
		int y = centroids.at<double>(j, 1);

		BigLabels[sumOfBigLables][0] = x;
		BigLabels[sumOfBigLables][1] = y;
		BigLabels[sumOfBigLables][2] = width;
		BigLabels[sumOfBigLables][3] = height;
		sumOfBigLables++;

		/*circle(img_color, Point(x, y), 10, Scalar(255, 0, 0), 1);

		rectangle(img_color, Point(left, top), Point(left + width, top + height),
			Scalar(0, 0, 255), 1);

		putText(img_color, to_string(sumOfBigLables), Point(left + 20, top + 20), FONT_HERSHEY_SIMPLEX,
			1, Scalar(255, 0, 0), 2);
	*/
	}
	connected = false;
	if (sumOfBigLables > 4)
		sumOfBigLables = 4;
	//4�� �̻��̸� 4�������� �˻�
	for (int i = 0; i < sumOfBigLables; i++) {
		for (int j = i + 1; j < sumOfBigLables; j++) {
			if ((BigLabels[i][1] >= BigLabels[j][1] - (BigLabels[j][3] / 2)) &&
				(BigLabels[i][1] <= BigLabels[j][1] + (BigLabels[j][3] / 2))) {//����ã�Ҿ�
				connectedLabels[0][0] = BigLabels[i][0];
				connectedLabels[0][1] = BigLabels[i][1];
				connectedLabels[0][2] = BigLabels[i][2];
				connectedLabels[0][3] = BigLabels[i][3];
				connectedLabels[1][0] = BigLabels[j][0];
				connectedLabels[1][1] = BigLabels[j][1];
				connectedLabels[1][2] = BigLabels[j][2];
				connectedLabels[1][3] = BigLabels[j][3];
				//
				circle(img_color, Point(connectedLabels[0][0], connectedLabels[0][1]), 10, Scalar(0, 255, 0), 1);
				circle(img_color, Point(connectedLabels[1][0], connectedLabels[1][1]), 10, Scalar(0, 255, 0), 1);
				line(img_color,
					Point(connectedLabels[0][0], connectedLabels[0][1]),
					Point(connectedLabels[1][0], connectedLabels[1][1]),
					Scalar(0, 0, 255), 2);
				connected = true;//���������� true�� ����
				i = sumOfBigLables - 1;//������
				j = sumOfBigLables - 1;//"
			}
		}
	}
	namedWindow("Eye Labeling Image", WINDOW_AUTOSIZE);             // Create a window for display
	imshow("Eye Labeling Image", img_color);                        // Show our image inside it

	for (int i = 0; i < numOfLables; i++)
		delete[] BigLabels[i];
	delete[] BigLabels;
	//�޸� ����

}
Mat DrawLabelingImage(Mat img_gray) {
	Mat img_color;
	Mat img_binary;
	threshold(img_gray, img_binary, 127, 255, THRESH_BINARY);
	cvtColor(img_gray, img_color, COLOR_GRAY2BGR);


	Mat img_labels, stats, centroids;
	int numOfLables = connectedComponentsWithStats(img_binary, img_labels,
		stats, centroids, 4, CV_32S);


	//�󺧸��� �̹����� Ư�� ���� �÷��� ǥ�����ֱ� �� ����� �Ⱦ���?��
	/*for (int y = 0; y<img_labels.rows; ++y) {

	int *label = img_labels.ptr<int>(y);
	Vec3b* pixel = img_color.ptr<Vec3b>(y);


	for (int x = 0; x < img_labels.cols; ++x) {


	if (label[x] == 3) {
	pixel[x][2] = 0;
	pixel[x][1] = 255;
	pixel[x][0] = 0;
	}
	}
	}
	*/

	//�󺧸� �� �̹����� ���� ���簢������ �ѷ��α� 
	int sumOfBigLables = 0;//�ʱⰪ -1
	for (int j = 1; j < numOfLables; j++) {
		int area = stats.at<int>(j, CC_STAT_AREA);
		int left = stats.at<int>(j, CC_STAT_LEFT);
		int top = stats.at<int>(j, CC_STAT_TOP);
		int width = stats.at<int>(j, CC_STAT_WIDTH);
		int height = stats.at<int>(j, CC_STAT_HEIGHT);
		if (width < 20 || height < 20) {
			continue;//����ũ�� ���ϸ� ����
		}
		//if (sumOfBigLables == -1) {
		//	sumOfBigLables++;
		//	continue;
		//}
		sumOfBigLables++;
		int x = centroids.at<double>(j, 0); //�߽���ǥ
		int y = centroids.at<double>(j, 1);

		circle(img_color, Point(x, y), 10, Scalar(255, 0, 0), 1);

		rectangle(img_color, Point(left, top), Point(left + width, top + height),
			Scalar(0, 0, 255), 1);

		putText(img_color, to_string(sumOfBigLables), Point(left + 20, top + 20), FONT_HERSHEY_SIMPLEX,
			1, Scalar(255, 0, 0), 2);
	}

	namedWindow("Labeling Image", WINDOW_AUTOSIZE);             // Create a window for display
	imshow("Labeling Image", img_color);                        // Show our image inside it
	return img_color;
}
Mat onSobel(Mat rgbImg) {
	float** hueBuffer;
	float** satuBuffer;
	float** intenBuffer;
	float** sobelBuffer;
	int imgHeight = rgbImg.rows;
	int imgWidth = rgbImg.cols;
	hueBuffer = new float*[imgHeight];//���Ǽ���
	satuBuffer = new float*[imgHeight];//���Ǽ���
	intenBuffer = new float*[imgHeight];//���Ǽ���
	sobelBuffer = new float*[imgHeight];//���Ǽ���
	for (int i = 0; i < imgHeight; i++) {
		hueBuffer[i] = new float[imgWidth];
		satuBuffer[i] = new float[imgWidth];
		intenBuffer[i] = new float[imgWidth];
		sobelBuffer[i] = new float[imgWidth];
	}
	for (int i = 0; i < imgHeight; i++) {
		for (int j = 0; j < imgWidth; j++) {
			float r = rgbImg.ptr<BYTE>(i, j)[2] / 255.0;
			float g = rgbImg.ptr<BYTE>(i, j)[1] / 255.0;
			float b = rgbImg.ptr<BYTE>(i, j)[0] / 255.0;

			intenBuffer[i][j] = (1.0 / 3.0)*(r + g + b);
			float minRGB;
			if (r > g) {
				minRGB = g;
				if (minRGB > b)
					minRGB = b;
			}
			else {
				minRGB = r;
				if (minRGB > b)
					minRGB = b;
			}
			satuBuffer[i][j] = 1.0 - (3.0 / (float)(r + g + b))*(minRGB);
			hueBuffer[i][j] = acos(((1.0 / 2.0)*((r - g) + (r - b))) / (sqrt((r - g)*(r - g) + (r - b)*(g - b))));
			hueBuffer[i][j] *= 57.29577951;
			if (b > g)
				hueBuffer[i][j] = 360 - hueBuffer[i][j];
			if (hueBuffer[i][j] < 0)
				hueBuffer[i][j] += 360;
			else if (hueBuffer[i][j] > 360)
				hueBuffer[i][j] -= 360;
		}
	}
	for (int i = 1; i < imgHeight - 1; i++) {
		for (int j = 1; j < imgWidth - 1; j++) {
			float Sx = intenBuffer[i - 1][j + 1] +
				2 * intenBuffer[i][j + 1] +
				intenBuffer[i + 1][j + 1] -
				intenBuffer[i - 1][j - 1] -
				2 * intenBuffer[i][j - 1] -
				intenBuffer[i + 1][j - 1];
			float Sy = intenBuffer[i - 1][j - 1] +
				2 * intenBuffer[i - 1][j] +
				intenBuffer[i - 1][j + 1] -
				intenBuffer[i + 1][j - 1] -
				2 * intenBuffer[i + 1][j] -
				intenBuffer[i + 1][j + 1];
			float Sxy = sqrt((float)Sx*Sx + (float)Sy*Sy)*255.0;
			if (Sxy < 80) {//96
				sobelBuffer[i][j] = 0;
			}
			//else if (Sxy >= 96 && Sxy <= 160) {
			//	sobelBuffer[i][j] = Sxy;
			//}
			else {
				sobelBuffer[i][j] = 255.0;
			}
		}
	}
	for (int i = 0; i < imgHeight; i++) {
		for (int j = 0; j < imgWidth; j++) {
			float H = hueBuffer[i][j];
			float S = satuBuffer[i][j];
			float I = sobelBuffer[i][j];
			float r = 0;
			float g = 0;
			float b = 0;
			if (0 <= H && H < 120) {
				float ceta = H*0.017453293;
				float ceta2 = (60.0 - H)*0.017453293;
				b = (1.0 / 3.0)*(float)(1 - S);
				r = (1.0 / 3.0)*(float)(1 + ((float)(S*cos(ceta)) / (float)(cos(ceta2))));
				g = 1 - (r + b);
			}
			else if (120 <= H&&H < 240) {
				H = H - 120;
				float ceta = H*0.017453293;
				float ceta2 = (60.0 - H)*0.017453293;
				g = (1.0 / 3.0)*(float)(1 + ((float)(S*cos(ceta)) / (float)(cos(ceta2))));
				r = (1.0 / 3.0)*(float)(1 - S);
				b = 1 - (r + g);
			}
			else if (240 <= H&&H <= 360) {
				H = H - 240;
				float ceta = H*0.017453293;
				float ceta2 = (60.0 - H)*0.017453293;
				b = (1.0 / 3.0)*(float)(1 + (float)((float)(S*cos(ceta)) / (float)(cos(ceta2))));
				g = (1.0 / 3.0)*(float)(1 - S);
				r = 1 - (g + b);
			}
			rgbImg.ptr<BYTE>(i, j)[2] = 3.0*r*I*255.0;
			rgbImg.ptr<BYTE>(i, j)[1] = 3.0*g*I*255.0;
			rgbImg.ptr<BYTE>(i, j)[0] = 3.0*b*I*255.0;
		}
	}
	for (int i = 0; i < imgHeight; i++) {
		delete[] hueBuffer[i];
		delete[] satuBuffer[i];
		delete[] intenBuffer[i];
		delete[] sobelBuffer[i];
	}
	delete[] hueBuffer;
	delete[] satuBuffer;
	delete[] intenBuffer;
	delete[] sobelBuffer;
	return rgbImg;
}

void CMFCApplication1_0605projectView::OnHandtracking() {



}


void On3ss(Mat frame) {
	//frame�� �޾Ƽ� 3ss�� ����

	if (frame.data == nullptr)
		AfxMessageBox("Error Video");

	//ũ�⸸ŭ ����
	aviHeight = frame.rows;
	aviWidth = frame.cols;
	aviBuffer1 = new RGBQUAD*[aviHeight];
	aviBuffer2 = new RGBQUAD*[aviHeight];

	for (int i = 0; i < aviHeight; i++)
	{
		aviBuffer1[i] = new RGBQUAD[aviWidth];
		aviBuffer2[i] = new RGBQUAD[aviWidth];
	}

	//3SS�� ���� �غ�
	FILE* file = fopen("HAND3SS.txt", "wt");
	CString value3SS;

	//frame ũ���� 16����1�� ��ϻ������ ����
	int hand_w = hand3ss.cols;
	int hand_h = hand3ss.rows;
	int block_w_n = aviWidth / hand_w;
	int block_h_n = aviHeight / hand_h;
	int value = 0;

	//namedWindow("img1", CV_WINDOW_AUTOSIZE);
	//imshow("img1", frame);

	/*
	��� �����Ӹ��� 3ss�� �ٷ� ����

	*/
	if (cnt % 2 == 0)
	{
		for (int i = 0; i < aviHeight; i++) {
			for (int j = 0; j < aviWidth; j++)
			{//BRG����
				aviBuffer2[i][j].rgbBlue = frame.ptr<BYTE>(i, j)[0];//���⿡ ù �������� ����Ǿ�� �� =>hand3ss
				aviBuffer2[i][j].rgbGreen = frame.ptr<BYTE>(i, j)[1];
				aviBuffer2[i][j].rgbRed = frame.ptr<BYTE>(i, j)[2];
				aviBuffer1[i][j].rgbBlue = 0;
				aviBuffer1[i][j].rgbGreen = 0;
				aviBuffer1[i][j].rgbRed = 0;
			}
		}
	}
	else
	{
		for (int i = 0; i < aviHeight; i++) {
			for (int j = 0; j < aviWidth; j++)
			{//BRG����
				aviBuffer1[i][j].rgbBlue = frame.ptr<BYTE>(i, j)[0];//���⿡ �ι�° �������� ����Ǿ�� ��=>hand3ss
				aviBuffer1[i][j].rgbGreen = frame.ptr<BYTE>(i, j)[1];
				aviBuffer1[i][j].rgbRed = frame.ptr<BYTE>(i, j)[2];
				aviBuffer2[i][j].rgbBlue = 0;
				aviBuffer2[i][j].rgbGreen = 0;
				aviBuffer2[i][j].rgbRed = 0;

			}
		}
	}


	//////��� �������� ���⼭���� ����
	if (cnt >= 1) {//3ss
		for (int i = 0; i < block_h_n; i++)
		{
			for (int j = 0; j < block_w_n; j++)
			{
				ArrValue av;
				av = On3SS(16 * (j + 1) - 8, 16 * (i + 1) - 8, av, 3, cnt);
				value3SS.Format(_T("(%3d, %3d) "), av.x_dif, av.y_dif);
				fprintf(file, value3SS);
			}

			fprintf(file, "\n");
		}
		value3SS.Format(_T("-%d & %d �������� ��Ǻ�ȭ��- \n\n"), cnt - 1, cnt);
		fprintf(file, value3SS);
	}
	//n�� 0�϶��� �� ����      //n�� 1�϶����� 3SS�����̰�, 
	//n�� ¦���϶����� 2->1 ������      //n�� Ȧ���϶����� 1->2 ��������
	cnt++;
	////////////////////////
	fclose(file);


	//AfxMessageBox("Completed");

}


ArrValue On3SS(int x, int y, ArrValue av, int number, int n) {
	int w = (int)pow(2, number); //ó�� ������ 3�̰ŵ�.
	int value = 0;
	double arr[9];
	double min = 800.0;
	int min_idx;
	for (int i = -1; i < 2; i++)
	{
		for (int j = -1; j < 2; j++)
		{
			int num = (j + 1) * 3 + (i + 1);
			int x2 = (i*w) + av.x_dif;
			int y2 = (j*w) + av.y_dif;
			//         if (!(x< 0 || x>aviWidth || y <0 || y>aviHeight) && !(x+x2<0 || x+x2>aviWidth|| y+y2<0||y+y2>aviHeight))
			if (!(x + x2 <= 0 || x + x2 >= aviWidth || y + y2 <= 0 || y + y2 >= aviHeight || x <= 0 || x >= aviWidth || y <= 0 || y >= aviHeight))
				arr[num] = calculate_subtraction(x, y, x2, y2, n);
			else
				arr[num] = 800;
		}
	}
	for (int i = 0; i < 9; i++)
	{
		if (arr[i] <= min && arr[i] >= 0)
		{
			min_idx = i;
			min = arr[i];
		}
	}
	/*CString str;
	str.Format("%lf", min);
	AfxMessageBox(str);
	*/if (min == 0) {
	//���� ������=>�̵����� ����
		min_idx = 4;

	}

	av.x_dif += (min_idx % 3 - 1)*w;
	av.y_dif += (min_idx / 3 - 1)*w;
	if (number > 1)
	{
		number--;
		av = On3SS(x, y, av, number, n);
	}
	return av;
}

double calculate_subtraction(int x, int y, int x_dif, int y_dif, int n) {
	int sum = 0;

	//��� �������� ������ ���, avibuffer2�� avibuffer1�� ����

	for (int i = -8; i < 8; i++)
	{
		for (int j = -8; j < 8; j++)
		{
			if (!((x + x_dif + i) <= 0 || (x + x_dif + i) >= aviWidth || (y + y_dif + j) <= 0 || (y + y_dif + j) >= aviHeight))
			{

				if (n % 2 == 1)//
				{
					sum += abs((int)aviBuffer1[y + j][x + i].rgbBlue - (int)aviBuffer2[y + y_dif + j][x + x_dif + i].rgbBlue);
					sum += abs((int)aviBuffer1[y + j][x + i].rgbGreen - (int)aviBuffer2[y + y_dif + j][x + x_dif + i].rgbGreen);
					sum += abs((int)aviBuffer1[y + j][x + i].rgbRed - (int)aviBuffer2[y + y_dif + j][x + x_dif + i].rgbRed);

				}
				else//¦����
				{
					sum += abs((int)aviBuffer1[y + y_dif + j][x + x_dif + i].rgbBlue - (int)aviBuffer2[y + j][x + i].rgbBlue);
					sum += abs((int)aviBuffer1[y + y_dif + j][x + x_dif + i].rgbGreen - (int)aviBuffer2[y + j][x + i].rgbGreen);
					sum += abs((int)aviBuffer1[y + y_dif + j][x + x_dif + i].rgbRed - (int)aviBuffer2[y + j][x + i].rgbRed);
				}
			}
			else {

				if (n % 2 == 1)//
				{
					sum += abs((int)aviBuffer1[y + j][x + i].rgbBlue - (int)aviBuffer2[y + y_dif + j][x + x_dif + i].rgbBlue);
					sum += abs((int)aviBuffer1[y + j][x + i].rgbGreen - (int)aviBuffer2[y + y_dif + j][x + x_dif + i].rgbGreen);
					sum += abs((int)aviBuffer1[y + j][x + i].rgbRed - (int)aviBuffer2[y + y_dif + j][x + x_dif + i].rgbRed);

				}
				else//¦����
				{

					if (n % 2 == 1) {
						sum += abs((int)aviBuffer1[y + j][x + i].rgbBlue);
						sum += abs((int)aviBuffer1[y + j][x + i].rgbGreen);
						sum += abs((int)aviBuffer1[y + j][x + i].rgbRed);
					}
					else
					{
						sum += abs((int)aviBuffer2[y + j][x + i].rgbBlue);
						sum += abs((int)aviBuffer2[y + j][x + i].rgbGreen);
						sum += abs((int)aviBuffer2[y + j][x + i].rgbRed);
					}
				}
			}
		}
	}

	return sum / 16.0 / 16;
}


void CMFCApplication1_0605projectView::OnHandlabeling()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.

	VideoCapture video(0);
	Mat tmpImg;//������, �Ŀ� ����� �������� ���� ���ϰ���
	Mat textureImg;//�ؽ���
	Mat textureMask;//�ؽ��� ����ũ
	Mat edgeImg;//����
	Mat result;//���
	Mat tmp;
	Mat newHand;
	int resultHandLabels[4] = { 0 };

	while (true) {//�ݺ�
		video >> tmp;
		video >> tmpImg;//������ �޾ƿ´�
		resize(tmpImg, tmpImg, Size(480, 320));
		try {
			namedWindow("original", WINDOW_AUTOSIZE);
			imshow("original", tmpImg);
			//ĸ���س� ���� ���
			cvtColor(tmpImg, textureImg, CV_BGR2YCrCb);
			//YCrCb �����̽��� ����
			inRange(textureImg, Scalar(0, 133, 77), Scalar(255, 173, 127), textureImg);
			//������ �Ǻλ� ������ ��, �� �ν� ����� grayscale �̹�����
			threshold(textureImg, textureMask, 127, 255, THRESH_BINARY);
			//����� ���� grayscale �̹����� threshold�ؼ� �̾Ƴ�, �Ŀ� ����ũ�� ����Ұ���
			//////////////////////////////////////////////////////////////////////////////////////////////
			//add(tmpImg, Scalar(0), textureImg, textureMask);
			//������ ���� ����� ����ũ�� �̿��� �Ǻ� �̿ܿ� �κ��� ���������� ĥ�ع���
			//add(~textureMask, Scalar(0), textureImg, ~textureMask);
			//�״㿡 �Ǻθ� ������� ĥ�� �ᱹ�� �Ǻ�=���, �׿�=������ ���� ����///////////////////////// �ٵ� �� ������ �ʿ䰡 ������;
			//////////////////////////////////////////////////////////////////////////////////////////////
			result = ~onGrayMedian(textureMask);
			//�׳� ����ũ �̾Ƴ��� ��ü�� �Ǻζ� ������ �����ִ� ���̳ʸ� �̹����� 
			//�̷��� �̰� �Ѱܼ� median �Լ��� �̿��� Ƣ�°� �����ش�

			edgeImg = onSobel(tmpImg);
			//ó������ ���ư� ���������� ���������ش�
			cvtColor(edgeImg, edgeImg, CV_BGR2GRAY);
			//�� ���� �̾Ƴ��� �׷��� �����Ϸ� �����Ѵ�
			threshold(edgeImg, edgeImg, 127, 255, THRESH_BINARY);
			//������Ȧ�� �ؼ� ����=���, �׿�=������ ���� ����
			edgeImg = EdgeImprove(~edgeImg);
			//������ �ִ� �ȼ��� ũ�� Ű���� ���� �� �̾������� ���ش�

			add(~edgeImg, Scalar(0), result, ~edgeImg);
			//���� ������ ��+�� ������ ���ļ� ��, �� ���еǵ��� �ؼ�
			///////////////input3ss = DrawLabelingImage(~result);

			//void DrawLabelingImageHand(Mat img_gray, int(*handLabels)[4], int& numOfHandLabels)
			//void DrawLabelingImageEye(Mat img_gray, int(*connectedLabels)[4], bool& connected)
			//AfxMessageBox(str);

			int eyeLabels[2][4] = { 0 };
			bool successOfEyeLabels = false;
			DrawLabelingImageEye(result, eyeLabels, successOfEyeLabels);
			if (successOfEyeLabels == false) {
				allowCnt++;
				if (allowCnt >= 3) {
					//////////////////////////////////


				//	AfxMessageBox(_T("�������� �Ǵ��Ա���°��� �� ����"));
				}
			}
			else {//Ŀ�ؼ��� �־�
				if (allowCnt > 0) {
					allowCnt--;
				}
			}

			////AfxMessageBox(str);

			//int handLabels[4] = { 0 };
			//bool subtracted;
			//if (newHand.data != nullptr) {
			//	//���� ���� hand�� ���� �ʱ�ȭ ���ش�
			//	newHand.data = nullptr;
			//}
			//newHand = DrawLabelingImageHand(~result, handLabels, subtracted, resultHandLabels);

			////	if (resultHandLabels[3] == 0) {
			///*CString str;
			//str.Format("%d",handFlag);
			//AfxMessageBox(str);*/

			////rectangle(result, Point(resultHandLabels[0], resultHandLabels[1]), Point(resultHandLabels[0] + resultHandLabels[2], resultHandLabels[1] + resultHandLabels[3]),
			////	Scalar(255, 0, 0), 1);
			//namedWindow("����?", WINDOW_AUTOSIZE);             // Create a window for display
			//imshow("����?", hand3ss);                       // Show our image inside it
			//namedWindow("����!", WINDOW_AUTOSIZE);             // Create a window for display
			//imshow("����!", newHand);                       // Show our image inside it
			//											  //	}//��� ���ó�� �������� BGR��(hand3ss)

			//if (hand3ss.data != nullptr) {
			//	//���⼭���� �ѹ� ���� ������, ������ frame��  newHand�� ������ 3ss�� �Ѵ�
			//	//���� ��Ǻ��Ͱ��� ���ġ�� �Ѵ°�� �޽��� �ڽ�
			//	On3ss(newHand);
			//}

		}
		catch (Exception& e) {
			cerr << "ó���� �����߱⿡, �ش� �������� �����մϴ�." << e.err << endl;
		}

		if (waitKey(27) == 27)
			break;
	}//while loop finish

	 //�޸� ����
	video.release();
	tmpImg.release();
	textureImg.release();
	textureMask.release();
	edgeImg.release();
	result.release();
	destroyAllWindows();
}


void CMFCApplication1_0605projectView::OnBackgroundsubway()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CFileDialog dlg(TRUE, ".avi", NULL, NULL, "AVI File (*.avi)|*.avi||");
	if (IDOK != dlg.DoModal())
		return;
	// Init background substractor
	CString cfilename = dlg.GetPathName();
	CT2CA strAtl(cfilename);
	String filename(strAtl);

	cv::VideoCapture Capture;
	Capture.open(filename);
	if (!Capture.isOpened())
		AfxMessageBox("Error Video");

	Mat img = imread("lotte.PNG", IMREAD_COLOR);
	resize(img, img, Size(480, 320));
	//
	VideoCapture video(0);
	Mat tmpImg;//������, �Ŀ� ����� �������� ���� ���ϰ���
	Mat textureImg;//�ؽ���
	Mat textureMask;//�ؽ��� ����ũ
	Mat edgeImg;//����
	Mat result;//���
	while (true) {//�ݺ�
		video >> tmpImg;//������ �޾ƿ´�
		resize(tmpImg, tmpImg, Size(480, 320));
		try {
			namedWindow("original", WINDOW_AUTOSIZE);
			imshow("original", tmpImg);
			//ĸ���س� ���� ���
			cvtColor(tmpImg, textureImg, CV_BGR2HSV);
			//YCrCb �����̽��� ����

			//inRange(textureImg, Scalar(0, 133, 77), Scalar(255, 173, 127), textureImg);

			//inRange(textureImg, Scalar(124, 42, 26), Scalar(173, 102, 97), textureImg);

			inRange(textureImg, Scalar(65, 60, 60), Scalar(80, 255, 255), textureImg);

			//������ �Ǻλ� ������ ��, �� �ν� ����� grayscale �̹�����
			threshold(textureImg, textureMask, 200, 255, THRESH_BINARY);
			//����� ���� grayscale �̹����� threshold�ؼ� �̾Ƴ�, �Ŀ� ����ũ�� ����Ұ���
			//////////////////////////////////////////////////////////////////////////////////////////////
			add(tmpImg, Scalar(0), textureImg, ~textureMask);
			//
			Mat backGround;
			add(img, Scalar(0), backGround, textureMask);
			add(backGround, textureImg, textureImg);
			namedWindow("backGroundChange", WINDOW_AUTOSIZE);
			imshow("backGroundChange", textureImg);
			//������ ���� ����� ����ũ�� �̿��� �Ǻ� �̿ܿ� �κ��� ���������� ĥ�ع���
			//add(~textureMask, Scalar(0), textureImg, ~textureMask);
			//�״㿡 �Ǻθ� ������� ĥ�� �ᱹ�� �Ǻ�=���, �׿�=������ ���� ����///////////////////////// �ٵ� �� ������ �ʿ䰡 ������;
			//////////////////////////////////////////////////////////////////////////////////////////////
			//result = ~onGrayMedian(textureMask);
			//�׳� ����ũ �̾Ƴ��� ��ü�� �Ǻζ� ������ �����ִ� ���̳ʸ� �̹����� 
			//�̷��� �̰� �Ѱܼ� median �Լ��� �̿��� Ƣ�°� �����ش�

			//edgeImg = onSobel(tmpImg);
			//ó������ ���ư� ���������� ���������ش�
			//cvtColor(edgeImg, edgeImg, CV_BGR2GRAY);
			//�� ���� �̾Ƴ��� �׷��� �����Ϸ� �����Ѵ�
			//threshold(edgeImg, edgeImg, 127, 255, THRESH_BINARY);
			//������Ȧ�� �ؼ� ����=���, �׿�=������ ���� ����
			//edgeImg = EdgeImprove(~edgeImg);
			//������ �ִ� �ȼ��� ũ�� Ű���� ���� �� �̾������� ���ش�

			//add(~edgeImg, Scalar(0), result, ~edgeImg);
			//���� ������ ��+�� ������ ���ļ� ��, �� ���еǵ��� �ؼ�
			///////////////input3ss = DrawLabelingImage(~result);
			//input3ss = DrawLabelingImage(~result);
			//�󺧵� �̾Ƴ���, �ڽ�ġ�� �׷��� �� �Լ� �ȿ��� �˾Ƽ� ���� ��
			/*namedWindow("sibal13", WINDOW_AUTOSIZE);
			imshow("sibal13", tmpImg);*/

			//cvtColor(input3ss, input3ss, CV_GRAY2BGR);

			//On3ss(tmp);
		}
		catch (Exception& e) {
			cerr << "ó���� �����߱⿡, �ش� �������� �����մϴ�." << e.err << endl;
		}

		if (waitKey(27) == 27)
			break;
	}//while loop finish

	 //�޸� ����
	video.release();
	tmpImg.release();
	textureImg.release();
	textureMask.release();
	edgeImg.release();
	result.release();
	destroyAllWindows();
}


