
// MFCApplication1_0605projectView.h : CMFCApplication1_0605projectView Ŭ������ �������̽�
//

#pragma once
class CMFCApplication1_0605projectView : public CView
{
protected: // serialization������ ��������ϴ�.
	CMFCApplication1_0605projectView();
	DECLARE_DYNCREATE(CMFCApplication1_0605projectView)

// Ư���Դϴ�.
public:
	CMFCApplication1_0605projectDoc* GetDocument() const;

// �۾��Դϴ�.
public:

// �������Դϴ�.
public:
	virtual void OnDraw(CDC* pDC);  // �� �並 �׸��� ���� �����ǵǾ����ϴ�.
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:
	virtual BOOL OnPreparePrinting(CPrintInfo* pInfo);
	virtual void OnBeginPrinting(CDC* pDC, CPrintInfo* pInfo);
	virtual void OnEndPrinting(CDC* pDC, CPrintInfo* pInfo);


// �����Դϴ�.
public:
	virtual ~CMFCApplication1_0605projectView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// ������ �޽��� �� �Լ�
protected:
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnLoadavi();
	afx_msg void OnBackgrounddetection();
	afx_msg void OnHanddetection();
//	Mat getHandMask1(const Mat& image, int minCr = 128, int maxCr = 170, int minCb = 73, int maxCb = 158);
	afx_msg void OnFacedetection();
	afx_msg void OnEdgedetection();
	afx_msg void OnLabeling();
	afx_msg void OnHandtracking();
	afx_msg void OnHandlabeling();
	afx_msg void OnBackgroundsubway();
};

#ifndef _DEBUG  // MFCApplication1_0605projectView.cpp�� ����� ����
inline CMFCApplication1_0605projectDoc* CMFCApplication1_0605projectView::GetDocument() const
   { return reinterpret_cast<CMFCApplication1_0605projectDoc*>(m_pDocument); }
#endif

