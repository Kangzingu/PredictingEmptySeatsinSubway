
// MFCApplication1_0605projectView.h : CMFCApplication1_0605projectView 클래스의 인터페이스
//

#pragma once
class CMFCApplication1_0605projectView : public CView
{
protected: // serialization에서만 만들어집니다.
	CMFCApplication1_0605projectView();
	DECLARE_DYNCREATE(CMFCApplication1_0605projectView)

// 특성입니다.
public:
	CMFCApplication1_0605projectDoc* GetDocument() const;

// 작업입니다.
public:

// 재정의입니다.
public:
	virtual void OnDraw(CDC* pDC);  // 이 뷰를 그리기 위해 재정의되었습니다.
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:
	virtual BOOL OnPreparePrinting(CPrintInfo* pInfo);
	virtual void OnBeginPrinting(CDC* pDC, CPrintInfo* pInfo);
	virtual void OnEndPrinting(CDC* pDC, CPrintInfo* pInfo);


// 구현입니다.
public:
	virtual ~CMFCApplication1_0605projectView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// 생성된 메시지 맵 함수
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

#ifndef _DEBUG  // MFCApplication1_0605projectView.cpp의 디버그 버전
inline CMFCApplication1_0605projectDoc* CMFCApplication1_0605projectView::GetDocument() const
   { return reinterpret_cast<CMFCApplication1_0605projectDoc*>(m_pDocument); }
#endif

