/****** Object:  StoredProcedure [dbo].[getpydata]    Script Date: 5/27/2020 8:23:29 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
-- =============================================
-- Author:		Sam Hartzog
-- Create date: 5/20/20
-- Description:	Get data for python
-- =============================================
CREATE PROCEDURE [dbo].[getpydata]
AS
BEGIN
	SELECT CASE nSex WHEN 1 THEN '0' ELSE '1' END 
      + CASE nPnemonia WHEN 1 THEN '1' ELSE '0' END
      + CASE nPregnancy WHEN 1 THEN '1' ELSE '0' END
      + CASE nDiabetes WHEN 1 THEN '1' ELSE '0' END
      + CASE nAthsma WHEN 1 THEN '1' ELSE '0' END
      + CASE nImmunosuppressed WHEN 1 THEN '1' ELSE '0' END
      + CASE nHypertension WHEN 1 THEN '1' ELSE '0' END
      + CASE nOtherDisease WHEN 1 THEN '1' ELSE '0' END
      + CASE nCardiovascularDisease WHEN 1 THEN '1' ELSE '0' END
      + CASE nObesity WHEN 1 THEN '1' ELSE '0' END
      + CASE nKidneyDisease WHEN 1 THEN '1' ELSE '0' END
      + CASE nTobbacco WHEN 1 THEN '1' ELSE '0' END
      + CASE nContactWithAnotherCase WHEN 1 THEN '1' ELSE '0' END
      + CASE nCOPD WHEN 1 THEN '1' ELSE '0' END
      + CASE WHEN dtDeath = '1970-01-01 00:00:00.000' AND dtSymptoms < '2020-04-15 00:00:00.000' THEN '1' ELSE '0' END
	  AS binData,
	  nAge as Age,
	  CASE nTypeOfCare WHEN 2 THEN '1' ELSE '0' END
      + CASE WHEN (nICU = 1 OR nIntubated = 1) THEN '1' ELSE '0' END
      + CASE dtDeath WHEN '1970-01-01 00:00:00.000' THEN '0' ELSE '1' END
	  AS labels
  FROM [dbo].[MX_CaseData]
  WHERE nTestResult = 1
END


