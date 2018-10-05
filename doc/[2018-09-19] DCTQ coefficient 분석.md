[2018-09-19] DCT/Q coefficient 분석
======

## TEncSearch::xEstimateResidualQT
여기에서 DCT/Q coefficient 는 다음에서 알 수 있다.

- Line 8756 - 9404 
~~~cpp
    const UInt uiNumCoeffPerAbsPartIdxIncrement = pcCU->getSlice()->getSPS()->getMaxCUWidth() * pcCU->getSlice()->getSPS()->getMaxCUHeight() >> ( pcCU->getSlice()->getSPS()->getMaxCUDepth() << 1 );
    const UInt uiQTTempAccessLayer = pcCU->getSlice()->getSPS()->getQuadtreeTULog2MaxSize() - uiLog2TrSize;
    TCoeff *pcCoeffCurrY = m_ppcQTTempCoeffY [uiQTTempAccessLayer] +  uiNumCoeffPerAbsPartIdxIncrement * uiAbsPartIdx;
~~~

실제 DCT 수행은 여기에서 이루어진다.
- Line 8803

~~~cpp
m_pcTrQuant->transformNxN
~~~

실제 위 함수는 다음의 단계를 거치게 된다.

~~~mermaid
graph TD;
A[Void TComTrQuant::transformNxN]-- pcCoeff -->B[Void TComTrQuant::xQuant]
B-- rpcCoeff -->C[Void TComTrQuant::xRateDistOptQuant]
C-- pDes -->B
B-- rpcCoeff -->A
~~~

- 1 차로 살펴본 결과 예상대로, Quantization 결과는 정수로만 나타난다. 대체로 다음과 같다.  (16x16의 예)
~~~
trWidth : 16
trHeight: 16
  2   2   1   0  -1  -1  -1  -1  -1   0   0   0   0   0   0   0
 -2  -2  -1   0   0   1   1   1   1   1   0   0   0   0   0   0
  0   0   0   1   1   1   0   0  -1  -1   0   0   0   0   0   0
  0   0   0  -1   0   0   0   0   1   1   0   0   0   0   0   0
  1   1   0   0  -1   0  -1  -1   0   0   0   0   0   0   0   0
 -1  -1   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
~~~
- 위 코드는 TEncSearch Line 8828에 추가 되어 있다.
~~~cpp
#if ETRI_AI_DUAL_QUNATIZE
		if (trWidth == 16){
		printf("trWidth : %d \n", trWidth);
		printf("trHeight: %d \n", trHeight);
		int test_sum = 0;
		for (int j = 0; j < trHeight; j++)
			for (int i = 0; i < trWidth; i++)
				test_sum += *(pcCoeffCurrY + j * trWidth + i);

		if (test_sum != 0){
			for (int j = 0; j < trHeight; j++){
				for (int i = 0; i < trWidth; i++)
					printf("%3d ", *(pcCoeffCurrY + j * trWidth + i));
				printf("\n");
			}
		}
		}
#endif
~~~

uiQTTempAccessLayer등을 사용하여 index를 주는 이유는 64x64가 Best CU로 선택되는 경우 TU Size가 32x32가 최대 이므로 4개의 TU로 구성되어져야 하기 때문이다. 그래서 uiNumCoeffPerAbsPartIdxIncrement = 16, uiAbsPartIdx=64, uiNumCoeffPerAbsPartIdxIncrement * uiAbsPartIdx : 1024 
로 index를 구해야 하는 것이다.
~~~
[LINE:7054 TEncSearch::ETRI_Check_AI_Info] uiQTTempAccessLayer : 0 
[LINE:7055 TEncSearch::ETRI_Check_AI_Info] uiNumCoeffPerAbsPartIdxIncrement: 16 
[LINE:7056 TEncSearch::ETRI_Check_AI_Info] uiAbsPartIdx        : 64 
[LINE:7057 TEncSearch::ETRI_Check_AI_Info] uiNumCoeffPerAbsPartIdxIncrement * uiAbsPartIdx : 1024 
[LINE:7058 TEncSearch::ETRI_Check_AI_Info] 
~~~
이를 통해 다음과 같은 코드로 coeff Pointer를 구하게 된다.

~~~cpp
		const UInt uiNumCoeffPerAbsPartIdxIncrement = pcCU->getSlice()->getSPS()->getMaxCUWidth() * pcCU->getSlice()->getSPS()->getMaxCUHeight() >> ( pcCU->getSlice()->getSPS()->getMaxCUDepth() << 1 );
		const UInt uiQTTempAccessLayer = pcCU->getSlice()->getSPS()->getQuadtreeTULog2MaxSize() - uiLog2TrSize;
		TCoeff *pcCoeffCurrY = m_ppcQTTempCoeffY [uiQTTempAccessLayer] +  uiNumCoeffPerAbsPartIdxIncrement * uiAbsPartIdx;
		TCoeff *pcCoeffCurrU = m_ppcQTTempCoeffCb[uiQTTempAccessLayer] + (uiNumCoeffPerAbsPartIdxIncrement * uiAbsPartIdx>>2);
		TCoeff *pcCoeffCurrV = m_ppcQTTempCoeffCr[uiQTTempAccessLayer] + (uiNumCoeffPerAbsPartIdxIncrement * uiAbsPartIdx>>2);
~~~

## Testbench 작성

### ETRI_CodeSubBlock 의 경우 
해당 함수는 xEstimateResidualQT 함수를 다시 부르는 Recursive 형태의 함수라는 점이다. 
따라서 em_DCTQData 와 같은 전역적 Struct가 있으면 이를 제대로 사용할 수 없다는 치명적인 문제점이 있다. 

본 코드 정리에서는 그래서, 재귀함수가 사용되지 않는 부분은 전역적 함수를 사용하지만, 그렇지 않은 부분은 Local Struct를 사용하여 재귀함수에 의해 전역 구조체 변수값이 변하는 경우를 막고자 한다. 

#### 재귀코드의 함수화에서는 논리의 문제에 유의하여야 한다.
만일 다음과 같은 코드가 있다고 하자.
~~~cpp
void func(void arg1, void &arg2)
{
    process_1;
    
    if (condition1)
    {
        process_2;
        if (condition2)
        {
            return;
        }
     }
}
~~~
이러한 형태의 코드를 다음과 같이 정리할 때는 **return** 부분을 매우 세심하게 살펴보아야 한다.

## DCTQ/IQIDCT 코드 정리 완료

다음 코드는 HM에서 수행하는 DCTQ/IDCTIQ를 수행하기 위한 TEncSearch::xEstimateResidualQT 함수를정리한 코드이다. 

~~~cpp
Void TEncSearch::xEstimateResidualQT( TComDataCU* pcCU, UInt uiQuadrant, UInt uiAbsPartIdx, UInt absTUPartIdx, TComYuv* pcResi, const UInt uiDepth, Double &rdCost, UInt &ruiBits, UInt &ruiDist, UInt *puiZeroDist )
{
	// code full block
	UInt uiSingleBits = 0, uiSingleDist = 0;
	UInt uiAbsSumY = 0, uiAbsSumU = 0, uiAbsSumV = 0;

	m_pcRDGoOnSbacCoder->store( m_pppcRDSbacCoder[ uiDepth ][ CI_QT_TRAFO_ROOT ] );

	ETRI_SubblockDataStruct _ITdata;
	ETRI_SetLocalParam(_ITdata, pcCU, uiAbsPartIdx, uiDepth, puiZeroDist);

	if( _ITdata.bCheckFull )
	{
		//--------------------------------------------------------------------------
		//  Processig Ready + Parameter Setting
		//--------------------------------------------------------------------------
		Int istdQP = pcCU->getQP(0);

		ETRI_ProcReadyDCTQ(pcCU, uiAbsPartIdx, uiDepth, _ITdata, istdQP);
		ETRI_SetParamYUV(setTransformSkipSubParts, 0, 0, 0, pcCU, uiAbsPartIdx, uiDepth,_ITdata);

		//--------------------------------------------------------------------------
		//  DCT/Quantization  Processing for Luma  and Chroma
		//		return value of the function 			
		//		em_DCTQData.uiAbsSumY    /U /V;
		//		em_DCTQData.pcCoeffCurrY /U /V;
		//--------------------------------------------------------------------------
		ETRI_DCTQunatization(pcCU, pcResi, uiAbsPartIdx, absTUPartIdx, uiDepth, istdQP);
		
#if ETRI_AI_DUAL_QUNATIZE
		//--------------------------------------------------------------------------
		//	A.I. Processing 
		//--------------------------------------------------------------------------
		ETRI_Check_AI_Info(pcCU, uiAbsPartIdx, istdQP, false);
#endif
		//--------------------------------------------------------------------------
		//	Evaluation of Bits for Luma and Chroma 
		//		return value of the function 			
		//		em_DCTQData.uiSingleBitsY / U  /V;
		//--------------------------------------------------------------------------
		ETRI_EvaluationBits(pcCU, uiAbsPartIdx, uiDepth);
		::memset( m_pTempPel, 0, sizeof( Pel ) * em_DCTQData.uiNumSamplesLuma ); // not necessary needed for inside of recursion (only at the beginning)

		//--------------------------------------------------------------------------		//	Inverse Quantization/Inverse DCT for Luma and Chroma 
		//		return value of the function 			
		//		uiSingleBits = _ITdata.uiSingleBits;
		//		uiSingleDist = _ITdata.uiSingleDist;
		//--------------------------------------------------------------------------		ETRI_IQIDCTSingleModule(pcCU, pcResi, uiAbsPartIdx, absTUPartIdx, uiDepth, _ITdata, istdQP);
		uiSingleBits 	= _ITdata.uiSingleBits;
		uiSingleDist 	= _ITdata.uiSingleDist;
		uiAbsSumY 	= em_DCTQData.uiAbsSumY;
		uiAbsSumU 	= em_DCTQData.uiAbsSumU;
		uiAbsSumV 	= em_DCTQData.uiAbsSumV;

	}

	//------------------------------------------------------------------------------	//	End of if bCheckFull 
	//------------------------------------------------------------------------------
	// code sub-blocks
	if (ETRI_CodeSubBlock(pcCU, pcResi, uiAbsPartIdx, uiDepth, _ITdata, rdCost, ruiBits, ruiDist)) return;

	rdCost += _ITdata.dSingleCost;
	ruiBits += uiSingleBits;
	ruiDist += uiSingleDist;

	pcCU->setTrIdxSubParts( _ITdata.uiTrMode, uiAbsPartIdx, uiDepth );
	ETRI_SetParamYUV(setCbfSubParts, (uiAbsSumY ? _ITdata.uiSetCbf : 0), (uiAbsSumU ? _ITdata.uiSetCbf : 0), (uiAbsSumV ? _ITdata.uiSetCbf : 0), pcCU,uiAbsPartIdx,uiDepth,_ITdata);

}
~~~

간단히 정리하면 다음과 같다.

~~~mermaid
graph TD
A[Start \ ETRI_SetLocalParam] --> B{_ITdata.bCheckFull}
B --> C[ETRI_ProcReadyDCTQ]
C--> D[ETRI_DCTQunatization]
D--> E[ETRI_Check_AI_Info]
E--> F[ETRI_EvaluationBits]
F--> G{ETRI_CodeSubBlock}
G--> H[Final Processing]
H-->I[return]
B-->G
G-->I
~~~

## Quantization Parameter Update 방식
함수 Void TComTrQuant::xQuant 의 다음 코드를 참조한다.  (Line 7247 - 7282)
~~~cpp
	Bool bRDOQ = false;

	QpParam cQpBase;
	Int iQpBase = pcCU->getSlice()->getSliceQpBase();

	Int qpScaled;
	Int qpBDOffset = (eTType == TEXT_LUMA) ? pcCU->getSlice()->getSPS()->getQpBDOffsetY() : pcCU->getSlice()->getSPS()->getQpBDOffsetC();

	if (eTType == TEXT_LUMA)
	{
		qpScaled = iQpBase + qpBDOffset;
	}
	else
	{
		Int chromaQPOffset;
		if (eTType == TEXT_CHROMA_U)
		{
			chromaQPOffset = pcCU->getSlice()->getPPS()->getChromaCbQpOffset() + pcCU->getSlice()->getSliceQpDeltaCb();
		}
		else
		{
			chromaQPOffset = pcCU->getSlice()->getPPS()->getChromaCrQpOffset() + pcCU->getSlice()->getSliceQpDeltaCr();
		}
		iQpBase = iQpBase + chromaQPOffset;

		qpScaled = Clip3(-qpBDOffset, 57, iQpBase);
		if (qpScaled < 0)
		{
			qpScaled = qpScaled + qpBDOffset;
		}
		else
		{
			qpScaled = g_aucChromaScale[qpScaled] + qpBDOffset;
		}
	}
	cQpBase.setQpParam(qpScaled);
~~~

여기서 맨 마지막 함수 cQpBase.setQpParam(qpScaled) 를 통해 Update를 하게 된다. 
class QpParam은 다음과 같다.

~~~cpp
class QpParam
{
public:
  QpParam();
  
  Int m_iQP;
  Int m_iPer;
  Int m_iRem;
  
public:
  Int m_iBits;
    
  Void setQpParam( Int qpScaled )
  {
    m_iQP   = qpScaled;
    m_iPer  = qpScaled / 6;
    m_iRem  = qpScaled % 6;
    m_iBits = QP_BITS + m_iPer;
  }
  
  Void clear()
  {
    m_iQP   = 0;
    m_iPer  = 0;
    m_iRem  = 0;
    m_iBits = 0;
  }
  
  
  Int per()   const { return m_iPer; }
  Int rem()   const { return m_iRem; }
  Int bits()  const { return m_iBits; }
  
  Int qp() {return m_iQP;}
}; // END CLASS DEFINITION QpParam
~~~


