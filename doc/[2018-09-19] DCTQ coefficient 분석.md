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




