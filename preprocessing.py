import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from datetime import datetime
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def consistencyCheck(row, ageLower = 0, ageHigher = 120, maxDailyHr = 18, e=1e-8, verbose=False):
    result = True
    if row['Age'] < ageLower or row['Age'] > ageHigher:
        if verbose:
            print(row['ClaimNumber'] + " Invalid Age: " + str(row['Age']) )
        result = False
    if row['Gender'] != 'M' and row['Gender'] != 'F':
        if verbose:
            print(row['ClaimNumber'] + " Minority Gender: " + str(row['Gender']))
        result = False
    if row['MaritalStatus'] not in ['M','S','U']:
        if verbose:
            print(row['ClaimNumber'] + " Invalid Marital Status: " + str(row['MaritalStatus']))
        result = False
    if row['DependentChildren'] < -e:
        if verbose:
            print(row['ClaimNumber'] + " Negative Number of Children: " + str(row['DependentChildren']))
        result = False
    if row['DependentsOther'] < -e:
        if verbose:
            print(row['ClaimNumber'] + " Negative Number of Other Dependents" + str(row['DependentsOther']))
        result = False
    if row['PartTimeFullTime'] not in ['P','F']:
        if verbose:
            print(row['ClaimNumber'] + " Invalid Job Type (Part Time / Full Time): " + str(row['PartTimeFullTime']))
        result = False
    if row['HoursWorkedPerWeek'] < e:
        if verbose:
            print(row['ClaimNumber'] + " Zero or Negative Working Hours: " + str(row['HoursWorkedPerWeek']))
        result = False
    if row['DaysWorkedPerWeek'] < 1 or row['DaysWorkedPerWeek'] > 7:
        if verbose:
            print(row['ClaimNumber'] + " Invalid Days Worked: " + str(row['DaysWorkedPerWeek']))
        result = False
    if row['HoursWorkedPerWeek'] > maxDailyHr * row['DaysWorkedPerWeek']:
        if verbose:
            print(row['ClaimNumber'] + " Inconsistent Working Hours And Days: " + str(np.round(row['HoursWorkedPerWeek']/row['DaysWorkedPerWeek'],2)))
        result = False
    return result

def dfConsistencyCheck(df, verbose):
    #L = len(df)
    result = []
    for i in df.index:
        result.append(consistencyCheck(df.loc[i], verbose = verbose))
    return result

def day2Report(strTAccident, strTReport):
    strFmt = "%Y-%m-%dT%H:%M:%SZ"
    tRep = datetime.strptime(strTReport, strFmt)
    tAcc = datetime.strptime(strTAccident, strFmt)
    result = np.ceil((tRep - tAcc).total_seconds()/(24*60*60))
    #if result < 0:
    #    result = 0
    return result

def applyDay2Report(df):
    return day2Report(df['DateTimeOfAccident'], df['DateReported'])

def getYMH(strTAccident):
    strFmt = "%Y-%m-%dT%H:%M:%SZ"
    tAcc = datetime.strptime(strTAccident, strFmt)
    return tAcc.year, tAcc.month, tAcc.hour

def accidentTime(df):
    return getYMH(df['DateTimeOfAccident'])

def getKeywords(df, word2ignore = ['TO','AND','ON','OF','IN','UP','FROM','A','ONTO','OFF','DOWN','WITH','BOTH','OUT','FOR'], threshold = 1e-4, stemmer = SnowballStemmer('english')):
    descriptions = ' '.join(df['ClaimDescription']) #bottleneck, change to df.apply format
    bagOfWords = descriptions.split(' ')
    N = len(bagOfWords)
    frac = {}
    stem2word = {}
    for word in bagOfWords:
        if word not in word2ignore:
            stem = stemmer.stem(word)
            if stem in frac:
                frac[stem] += 1/N
            else:
                frac[stem] = 1/N
                stem2word[stem] = word
        stems = [stem for stem in frac if frac[stem] > threshold]
    return [stem2word[stem] for stem in stems]#, stems

def addDescFeature(df, word2ignore = ['TO','AND','ON','OF','IN','UP','FROM','A','ONTO','OFF','DOWN','WITH','BOTH','OUT','FOR'], threshold = 1e-4, stemmer = SnowballStemmer('english'), transform = None, keywords = None):
    if keywords is None:
        descriptions = ' '.join(df['ClaimDescription']) #bottleneck, change to df.apply format
        bagOfWords = descriptions.split(' ')
        #stemmer = SnowballStemmer('english')
        N = len(bagOfWords)
        frac = {}
        stem2word = {}
        for word in bagOfWords:
            if word not in word2ignore:
                stem = stemmer.stem(word)
                if stem in frac:
                    frac[stem] += 1/N
                else:
                    frac[stem] = 1/N
                    stem2word[stem] = word
        stems = [stem for stem in frac if frac[stem] > threshold]
    else:
        stems = []
        stem2word = {}
        for word in keywords:
            if word not in word2ignore:
                stem = stemmer.stem(word)
                if stem not in stems:
                    stems.append(stem)
                    stem2word[stem] = word
    X = wordCountMatrix(df['ClaimDescription'],stems,stemmer)
    if transform is not None:
        X = transform(X)
    colNames = ["KW:"+stem2word[stem] for stem in stems]
    df_word_count = pd.DataFrame(X,columns=colNames,index=df.index)
    df = pd.concat([df, df_word_count], axis=1)
    return df
    #for i in range(len(stems)):
    #    stem = stems[i]
    #    colName = "KW:"+stem2word[stem]
    #    df[colName] = pd.Series(X[:,i])
    #if keywords is None:
    #    return [stem2word[stem] for stem in stems]
    #TODO change to "apply" format

def wordCountMatrix(docs, stems, stemmer):
    d = {}
    col_names = []
    for stem in stems:
        if stem not in d:
            d[stem] = len(d)
            col_names.append(stem)
    m = len(docs)
    n = len(stems)
    X = np.zeros((m,n))
    c = 0
    for i in docs.index:
        bagOfWords = docs[i].split(' ')
        for word in bagOfWords:
            stem = stemmer.stem(word)
            if stem in d:
                X[c][d[stem]] += 1
        c += 1
    return X

def addNumericFeatures(df, dfref=None, smooth = False, order = None, quantities2deflate = ['HourlyWage','DailyWage','WeeklyWages','InitialIncurredClaimCost','UltimateIncurredClaimCost']):
    df['DayToReport'] = df.apply(applyDay2Report, axis=1)
    df[['Year','Month','Hour']] = df.apply(accidentTime, axis=1,result_type='expand')
    df['HourlyWage'] = df['WeeklyWages']/df['HoursWorkedPerWeek']
    df['DailyWage'] = df['WeeklyWages']/df['DaysWorkedPerWeek']
    df[['DescLenChar','DescLenWord']] = df.apply(applyDescLengths, axis=1, result_type='expand')
    df['GenderIsM'] = (df['Gender'] == 'M').astype(int)
    df['GenderIsF'] = (df['Gender'] == 'F').astype(int)
    df['MaritalIsM'] = (df['MaritalStatus'] == 'M').astype(int)
    df['MaritalIsS'] = (df['MaritalStatus'] == 'S').astype(int)
    df['MaritalIsU'] = (df['MaritalStatus'] == 'U').astype(int)
    df['FullTime'] = (df['PartTimeFullTime'] == 'F').astype(int)
    if dfref is None:
        dfref = df
    else:
        if 'Year' not in dfref:
            dfref[['Year','Month','Hour']] = dfref.apply(accidentTime, axis=1,result_type='expand')
        if 'HourlyWage' not in dfref:
            dfref['HourlyWage'] = dfref['WeeklyWages']/dfref['HoursWorkedPerWeek']
        if 'DailyWage' not in dfref:
            dfref['DailyWage'] = dfref['WeeklyWages']/dfref['DaysWorkedPerWeek']
    for quantity in quantities2deflate:
        deflate(df, dfref, quantity, smooth, order)
    return df

def addFeatures(df, dfref = None, word2ignore = ['TO','AND','ON','OF','IN','UP','FROM','A','ONTO','OFF','DOWN','WITH','BOTH','OUT','FOR'], threshold = 1e-4, stemmer = SnowballStemmer('english'), transform = None, keywords = None, smooth = False, order = None, quantities2deflate = ['HourlyWage','DailyWage','WeeklyWages','InitialIncurredClaimCost']):#,'UltimateIncurredClaimCost']):
    df['DayToReport'] = df.apply(applyDay2Report, axis=1)
    df[['Year','Month','Hour']] = df.apply(accidentTime, axis=1,result_type='expand')
    df['HourlyWage'] = df['WeeklyWages']/df['HoursWorkedPerWeek']
    df['DailyWage'] = df['WeeklyWages']/df['DaysWorkedPerWeek']
    df[['DescLenChar','DescLenWord']] = df.apply(applyDescLengths, axis=1, result_type='expand')
    df['GenderIsM'] = (df['Gender'] == 'M').astype(int)
    df['GenderIsF'] = (df['Gender'] == 'F').astype(int)
    df['MaritalIsM'] = (df['MaritalStatus'] == 'M').astype(int)
    df['MaritalIsS'] = (df['MaritalStatus'] == 'S').astype(int)
    df['MaritalIsU'] = (df['MaritalStatus'] == 'U').astype(int)
    df['FullTime'] = (df['PartTimeFullTime'] == 'F').astype(int)
    if dfref is None:
        dfref = df
    else:
        if 'Year' not in dfref:
            dfref[['Year','Month','Hour']] = dfref.apply(accidentTime, axis=1,result_type='expand')
        if 'HourlyWage' not in dfref:
            dfref['HourlyWage'] = dfref['WeeklyWages']/dfref['HoursWorkedPerWeek']
        if 'DailyWage' not in dfref:
            dfref['DailyWage'] = dfref['WeeklyWages']/dfref['DaysWorkedPerWeek']
    if keywords is None:
        keywords = getKeywords(dfref)
    df = addDescFeature(df, word2ignore, threshold, stemmer, transform, keywords)
    for quantity in quantities2deflate:
        deflate(df, dfref, quantity, smooth, order, verbose=True)
    return df


def removeInvalid(df, ageLower = 0, ageHigher = 120, e=1e-8, maxDailyHr = 18):
    df.loc[(df['Age']<ageLower) | (df['Age']>ageHigher), 'Age'] = np.nan
    df.loc[(df['Gender'] != 'M') & (df['Gender'] != 'F'), 'Gender'] = pd.NaT
    df.loc[(df['MaritalStatus'] != 'M') & (df['MaritalStatus'] != 'S') & (df['MaritalStatus'] != 'U'), 'MaritalStatus'] = pd.NaT
    df.loc[df['DependentChildren']<-e, 'DependentChildren'] = np.nan
    df.loc[df['DependentsOther']<-e, 'DependentsOther'] = np.nan
    df.loc[(df['PartTimeFullTime'] != 'P') & (df['PartTimeFullTime'] != 'F'), 'PartTimeFullTime'] = pd.NaT
    df.loc[(df['HoursWorkedPerWeek']<e) | (df['HoursWorkedPerWeek']>maxDailyHr*7), 'HoursWorkedPerWeek'] = np.nan
    df.loc[(df['DaysWorkedPerWeek']<1) | (df['DaysWorkedPerWeek']>7), 'DaysWorkedPerWeek'] = np.nan

def col2change(row, hrwg, dywg):
    year = row['Year']#.values[0]
    hr = row['HourlyWage']#.values[0]
    dy = row['DailyWage']#.values[0]
    qhr = len([w for w in hrwg[year] if w > hr])/len(hrwg[year])
    phr = min(qhr,1-qhr)
    qdy = len([w for w in dywg[year] if w > dy])/len(dywg[year])
    pdy = min(qdy,1-qdy)
    #print(row['HoursWorkedPerWeek'].values[0],row['DaysWorkedPerWeek'].values[0],phr, pdy)
    if phr > pdy:
        return ["DaysWorkedPerWeek", "DailyWage", "RealDailyWage"]
    else:
        return ["HoursWorkedPerWeek", "HourlyWage", "RealHourlyWage"]

def removeInconsistent(df, maxDailyHr, hrwg = None, dywg = None):
    #for i in df.index:
    #    if df.loc[i,'HoursWorkedPerWeek'] > maxDailyHr * df.loc[i,'DaysWorkedPerWeek']:
    #        df.loc[i,[col for col in col2change(df.loc[i], hrwg, dywg) if col in df]] = np.nan
    if hrwg is None and dywg is None:
        for i in df[df['HoursWorkedPerWeek'] > maxDailyHr * df['DaysWorkedPerWeek']].index:
            df.loc[i,[col for col in ["DaysWorkedPerWeek", "DailyWage", "RealDailyWage", "HoursWorkedPerWeek", "HourlyWage", "RealHourlyWage"] if col in df]] = np.nan
    else:
        for i in df[df['HoursWorkedPerWeek'] > maxDailyHr * df['DaysWorkedPerWeek']].index:
            df.loc[i,[col for col in col2change(df.loc[i], hrwg, dywg) if col in df]] = np.nan

def applyDescLengths(df):
    return len(df['ClaimDescription']), len((df['ClaimDescription']).split(' '))

def deflator(df, quantity, smooth = False, order=None, verbose = False, cumChange = False, initYear = None):
    byYear = df.groupby(['Year',])[quantity].apply(list)
    mean = byYear.apply(np.mean)
    if initYear is None:
    	initYear = min(byYear.index)
    if cumChange:
        base = mean[initYear]
        mean /= base
    if not smooth:
        return mean
    b = np.polyfit(byYear.index-initYear, mean, order)
    b /= b[-1]
    smoothed = np.polyval(b, byYear.index-initYear)
    data = {'Mean':mean.values, 'Smoothed':smoothed}
    result = pd.DataFrame(data, index = byYear.index)
    if verbose:
        std = byYear.apply(np.std)/byYear.apply(len)**0.5
        if cumChange:
            std /= base
        plt.plot(byYear.index, mean, 'k-')
        plt.plot(byYear.index, smoothed, 'r-')
        z = norm.ppf(1-0.05/2/len(byYear))
        plt.plot(byYear.index, mean+z*std,'g--')
        plt.plot(byYear.index, mean-z*std,'g--')
        plt.show()
        #plt.xlabel("year")
        #plt.ylabel(quantity)
    return result

def deflate(df, dfref, quantity, smooth = False, order = None, verbose = False):
    cumChg = deflator(dfref, quantity, smooth, order, verbose, cumChange = True)
    #print(verbose)
    if smooth:
        factor = cumChg['Smoothed'][df['Year']]
    else:
        factor = cumChg[df['Year']]
    #print(factor.head())
    factor.index = df.index#np.arange(len(df))
    df['Real' + quantity] = df[quantity] / factor
    #return factor

def inflationFactor(df, dfref, quantity, smooth = False, order = None, verbose = False):
    cumChg = deflator(dfref, quantity, smooth, order, verbose, cumChange = True)
    #print(deflator.head())
    if smooth:
        factor = cumChg['Smoothed'][df['Year']]
    else:
        factor = cumChg[df['Year']]
    return factor
    
def impute(df, dfref):
    df.loc[df['Age'].isna(),'Age'] = dfref['Age'].median()
    df.loc[df['DependentChildren'].isna(),'DependentChildren'] = dfref['DependentChildren'].median()
    df.loc[df['DependentsOther'].isna(),'DependentsOther'] = dfref['DependentsOther'].median()
    df.loc[df['WeeklyWages'].isna(),'WeeklyWages'] = dfref['WeeklyWages'].median()
    df.loc[df['HoursWorkedPerWeek'].isna(),'HoursWorkedPerWeek'] = dfref['HoursWorkedPerWeek'].median()
    df.loc[df['DaysWorkedPerWeek'].isna(),'DaysWorkedPerWeek'] = dfref['DaysWorkedPerWeek'].median()
