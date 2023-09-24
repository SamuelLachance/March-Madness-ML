

### import packages
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import datetime as dt
import scipy.stats
import sys
import openturns as ot
from IPython.display import display
import requests
from datetime import datetime
import yfinance as yf
import io
from yahooquery import Ticker, Screener
from sklearn.linear_model import LinearRegression
from bs4 import BeautifulSoup
import re
import json
from yahoo_fin import stock_info as si

pd.options.display.max_columns=1000
pd.options.display.max_rows= 200
pd.options.display.float_format = '{:,.3f}'.format

def data_frame_flattener(df_data):
    df=df_data.copy()
    try:
        df.columns=[' '.join(map(str,col)).strip() for col in df.columns.values]
    except:
        pass
    return(df)

def column_suffix_adder(df_data,
                        list_of_columns_to_add_suffix_on,
                        suffix):
    """Add specific siffix to specific columns"""
    df=df_data.copy()
    ### Add suffix or prefix to certain columns rename all columns
    new_names = [(i,i+suffix) for i in df[list_of_columns_to_add_suffix_on].columns.values]
    df.rename(columns = dict(new_names), inplace=True)
    return(df)


# ### Valuation functions


def dynamic_converger(current,
                      expected,
                      number_of_steps,
                      period_to_begin_to_converge):
    number_of_steps =  int(number_of_steps)
    period_to_begin_to_converge = int(period_to_begin_to_converge)
    def converger(current,
                expected,
                number_of_steps):
        values = np.linspace(current,expected,number_of_steps+1)
        return(values)

    array_phase1 = np.array([current]*(period_to_begin_to_converge-1))

    array_phase2 = converger(current,
                       expected,
                       number_of_steps-period_to_begin_to_converge)
    result= pd.Series(np.concatenate((array_phase1,array_phase2)))
    return(result)

def dynamic_converger_multiple_phase(growth_rates_for_each_cylce,
                                     length_of_each_cylce,
                                     convergance_periods):
    list_of_results = []
    for cycle in range(len(length_of_each_cylce)):
        result = dynamic_converger(current = growth_rates_for_each_cylce[cycle][0],
                        expected = growth_rates_for_each_cylce[cycle][1],
                        number_of_steps = length_of_each_cylce[cycle],
                        period_to_begin_to_converge = convergance_periods[cycle])
        list_of_results.append(result)
    return(pd.concat(list_of_results,ignore_index=True))


def revenue_projector_multi_phase(revenue_base,
                                  revenue_growth_rate_cycle1_begin,
                                  revenue_growth_rate_cycle1_end,
                                  revenue_growth_rate_cycle2_begin,
                                  revenue_growth_rate_cycle2_end,
                                  revenue_growth_rate_cycle3_begin,
                                  revenue_growth_rate_cycle3_end = 0.028,
                                  length_of_cylcle1=3,
                                  length_of_cylcle2=4,
                                  length_of_cylcle3=3,
                                  revenue_convergance_periods_cycle1 =1,
                                  revenue_convergance_periods_cycle2=1,
                                  revenue_convergance_periods_cycle3=1):
    projected_revenue_growth = dynamic_converger_multiple_phase(growth_rates_for_each_cylce= [[revenue_growth_rate_cycle1_begin,revenue_growth_rate_cycle1_end],
                                                               [revenue_growth_rate_cycle2_begin,revenue_growth_rate_cycle2_end],
                                                               [revenue_growth_rate_cycle3_begin,revenue_growth_rate_cycle3_end]],
                                     length_of_each_cylce=[length_of_cylcle1,length_of_cylcle2,length_of_cylcle3],
                                     convergance_periods=[revenue_convergance_periods_cycle1,
                                                          revenue_convergance_periods_cycle2,
                                                          revenue_convergance_periods_cycle3])
    ### Compute Cummulative revenue_growth
    projected_revenue_growth_cumulative = (1+projected_revenue_growth).cumprod()
    projected_revneues = revenue_base*projected_revenue_growth_cumulative
    return(projected_revneues,projected_revenue_growth)


# In[ ]:


def operating_margin_projector(current_operating_margin,
                               terminal_operating_margin,
                               valuation_interval_in_years=10,
                               year_operating_margin_begins_to_converge_to_terminal_operating_margin=5):
    projectd_operating_margin = dynamic_converger(current_operating_margin,
                                                  terminal_operating_margin,
                                                  valuation_interval_in_years,
                                                  year_operating_margin_begins_to_converge_to_terminal_operating_margin)
    return(projectd_operating_margin)


# In[ ]:


def tax_rate_projector(current_effective_tax_rate,
                      marginal_tax_rate,
                      valuation_interval_in_years=10,
                      year_effective_tax_rate_begin_to_converge_marginal_tax_rate=5):
    """Project tax rate during valuation Cylce"""
    projected_tax_rate = dynamic_converger(current_effective_tax_rate,
                                           marginal_tax_rate,
                                           valuation_interval_in_years,
                                           year_effective_tax_rate_begin_to_converge_marginal_tax_rate)
    return(projected_tax_rate)


# In[ ]:


def cost_of_capital_projector(unlevered_beta,
                              terminal_unlevered_beta,
                              current_pretax_cost_of_debt,
                              terminal_pretax_cost_of_debt,
                              equity_value,
                              debt_value,
                              marginal_tax_rate=.21,
                              risk_free_rate=0.015,
                              ERP=0.055,
                              valuation_interval_in_years=10,
                              year_beta_begins_to_converge_to_terminal_beta=5,
                              year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt=5):
    """Project Cost of Capiatal during valuation Cylce"""
    ### Compute Beta During Valuatio Cycle
    ### Company Levered Beta  = Unlevered beta * (1 + (1- tax rate) (Debt/Equity))
    company_beta = unlevered_beta * (1+(1-marginal_tax_rate)*(debt_value/equity_value))
    terminal_beta = terminal_unlevered_beta * (1+(1-marginal_tax_rate)*(debt_value/equity_value))
    beta_druing_valution_cycle = dynamic_converger(company_beta,
                                                   terminal_beta,
                                                   valuation_interval_in_years,
                                                   year_beta_begins_to_converge_to_terminal_beta)
    ### Compute Pre Tax Cost Of debt During Valuation Cycle
    pre_tax_cost_of_debt_during_valution_cycle = dynamic_converger(current_pretax_cost_of_debt,
                                                                   terminal_pretax_cost_of_debt,
                                                                   valuation_interval_in_years,
                                                                   year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt)

    total_capital = equity_value+debt_value
    equity_to_capital = equity_value/total_capital
    debt_to_capital = debt_value/total_capital
    after_tax_cost_of_debt_during_valution_cycle = pre_tax_cost_of_debt_during_valution_cycle*(1-marginal_tax_rate)
    cost_of_equity = risk_free_rate + (beta_druing_valution_cycle*ERP)
    cost_of_capital_during_valuatio_cycle = ((equity_to_capital*cost_of_equity)+
                                             (debt_to_capital*after_tax_cost_of_debt_during_valution_cycle))
    return(cost_of_capital_during_valuatio_cycle,beta_druing_valution_cycle,terminal_beta,cost_of_equity,after_tax_cost_of_debt_during_valution_cycle)

def revenue_growth_projector(revenue_growth_rate,
                             terminal_growth_rate=.028,
                             valuation_interval_in_years=10,
                             year_revenue_growth_begin_to_converge_to_terminal_growth_rate = 5):
    """Project revenue growth during valuation Cylce"""
    projected_revenue_growth = dynamic_converger(revenue_growth_rate,
                                                 terminal_growth_rate,
                                                 valuation_interval_in_years,
                                                 year_revenue_growth_begin_to_converge_to_terminal_growth_rate)
    return(projected_revenue_growth)


def revenue_projector(revenue_base,
                      revenue_growth_rate,
                      terminal_growth_rate,
                      valuation_interval_in_years,
                      year_revenue_growth_begin_to_converge_to_terminal_growth_rate):
    ### Estimate Revenue Growth
    projected_revenue_growth = revenue_growth_projector(revenue_growth_rate=revenue_growth_rate,
                                                        terminal_growth_rate = terminal_growth_rate,
                                                        valuation_interval_in_years=valuation_interval_in_years,
                                                        year_revenue_growth_begin_to_converge_to_terminal_growth_rate=year_revenue_growth_begin_to_converge_to_terminal_growth_rate)
    ### Compute Cummulative revenue_growth
    projected_revenue_growth_cumulative = (1+projected_revenue_growth).cumprod()
    projected_revneues = revenue_base*projected_revenue_growth_cumulative
    return(projected_revneues,projected_revenue_growth)


def sales_to_capital_projector(current_sales_to_capital_ratio,
                               terminal_sales_to_capital_ratio,
                               valuation_interval_in_years=10,
                               year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital=3):
    projectd_sales_to_capiatl = dynamic_converger(current_sales_to_capital_ratio,
                                                  terminal_sales_to_capital_ratio,
                                                  valuation_interval_in_years,
                                                  year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital)
    return(projectd_sales_to_capiatl)


def reinvestment_projector(revenue_base,
                           projected_revneues,
                           sales_to_capital_ratios,
                           asset_liquidation_during_negative_growth=0):
    reinvestment = (pd.concat([pd.Series(revenue_base),
                               projected_revneues],
                             ignore_index=False).diff().dropna()/sales_to_capital_ratios)
    reinvestment = reinvestment.where(reinvestment>0, (reinvestment*asset_liquidation_during_negative_growth))
    return(reinvestment)


def valuator_multi_phase(
    risk_free_rate,
    ERP,
    equity_value,
    debt_value,
    unlevered_beta,
    terminal_unlevered_beta,
    year_beta_begins_to_converge_to_terminal_beta,
    current_pretax_cost_of_debt,
    terminal_pretax_cost_of_debt,
    year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt,
    current_effective_tax_rate,
    marginal_tax_rate,
    year_effective_tax_rate_begin_to_converge_marginal_tax_rate,
    revenue_base,
    revenue_growth_rate_cycle1_begin,
    revenue_growth_rate_cycle1_end,
    revenue_growth_rate_cycle2_begin,
    revenue_growth_rate_cycle2_end,
    revenue_growth_rate_cycle3_begin,
    revenue_growth_rate_cycle3_end,
    revenue_convergance_periods_cycle1,
    revenue_convergance_periods_cycle2,
    revenue_convergance_periods_cycle3,
    length_of_cylcle1,
    length_of_cylcle2,
    length_of_cylcle3,
    current_sales_to_capital_ratio,
    terminal_sales_to_capital_ratio,
    year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital,
    current_operating_margin,
    terminal_operating_margin,
    year_operating_margin_begins_to_converge_to_terminal_operating_margin,
    additional_return_on_cost_of_capital_in_perpetuity=0.0,
    cash_and_non_operating_asset=0.0,
    asset_liquidation_during_negative_growth=0,
    current_invested_capital='implicit'):
  
    valuation_interval_in_years = int(length_of_cylcle1) + int(length_of_cylcle2) + int(length_of_cylcle3)
    terminal_growth_rate = revenue_growth_rate_cycle3_end
    ### Estimate Cost of Capital during the valution cycle
    projected_cost_of_capital, projected_beta , terminal_beta , projected_cost_of_equity , projected_after_tax_cost_of_debt = cost_of_capital_projector(unlevered_beta=unlevered_beta,
                                                          terminal_unlevered_beta=terminal_unlevered_beta,
                                                          current_pretax_cost_of_debt=current_pretax_cost_of_debt,
                                                          terminal_pretax_cost_of_debt=terminal_pretax_cost_of_debt,
                                                          equity_value=equity_value,
                                                          debt_value=debt_value,
                                                          marginal_tax_rate=marginal_tax_rate,
                                                          risk_free_rate=risk_free_rate,
                                                          ERP=ERP,
                                                          valuation_interval_in_years=valuation_interval_in_years,
                                                          year_beta_begins_to_converge_to_terminal_beta=year_beta_begins_to_converge_to_terminal_beta,
                                                          year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt=year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt)
    projected_cost_of_capital_cumulative= (1+projected_cost_of_capital).cumprod()
    projected_cost_of_equity_cumulative= (1+projected_cost_of_equity).cumprod()
    ### Estimate Future revnues, and growth

    projected_revneues,projected_revenue_growth = revenue_projector_multi_phase(revenue_base = revenue_base,
                                  revenue_growth_rate_cycle1_begin = revenue_growth_rate_cycle1_begin,
                                  revenue_growth_rate_cycle1_end = revenue_growth_rate_cycle1_end,
                                  revenue_growth_rate_cycle2_begin = revenue_growth_rate_cycle2_begin,
                                  revenue_growth_rate_cycle2_end = revenue_growth_rate_cycle2_end,
                                  revenue_growth_rate_cycle3_begin = revenue_growth_rate_cycle3_begin,
                                  revenue_growth_rate_cycle3_end = revenue_growth_rate_cycle3_end,
                                  length_of_cylcle1=length_of_cylcle1,
                                  length_of_cylcle2=length_of_cylcle2,
                                  length_of_cylcle3=length_of_cylcle3,
                                  revenue_convergance_periods_cycle1 = revenue_convergance_periods_cycle1,
                                  revenue_convergance_periods_cycle2 = revenue_convergance_periods_cycle2,
                                  revenue_convergance_periods_cycle3 = revenue_convergance_periods_cycle3)
    ### Estmimate tax rates
    projected_tax_rates = tax_rate_projector(current_effective_tax_rate=current_effective_tax_rate,
                                            marginal_tax_rate=marginal_tax_rate,
                                            valuation_interval_in_years=valuation_interval_in_years,
                                            year_effective_tax_rate_begin_to_converge_marginal_tax_rate=year_effective_tax_rate_begin_to_converge_marginal_tax_rate)
    ### Estimate slaes to capital ratio during valuation for reinvestment
    sales_to_capital_ratios = sales_to_capital_projector(current_sales_to_capital_ratio,
                               terminal_sales_to_capital_ratio,
                               valuation_interval_in_years=valuation_interval_in_years,
                               year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital=year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital)

    ### Estimate Reinvestemnt
    projected_reinvestment = reinvestment_projector(revenue_base=revenue_base,
                                                    projected_revneues = projected_revneues,
                                                    sales_to_capital_ratios=sales_to_capital_ratios,
                                                    asset_liquidation_during_negative_growth=asset_liquidation_during_negative_growth)

    ### Estimate invested Capital
    invested_capital = projected_reinvestment.copy()
    if current_invested_capital == 'implicit':
        current_invested_capital = revenue_base / current_sales_to_capital_ratio
    invested_capital[0] = invested_capital[0] + current_invested_capital
    invested_capital = invested_capital.cumsum()

    ### Operating Margin
    projected_operating_margins = operating_margin_projector(current_operating_margin,
                                                            terminal_operating_margin,
                                                            valuation_interval_in_years=valuation_interval_in_years,
                                                            year_operating_margin_begins_to_converge_to_terminal_operating_margin=year_operating_margin_begins_to_converge_to_terminal_operating_margin)
    ###EBIT
    projected_operating_income = projected_revneues * projected_operating_margins
    ### After Tax EBIT (EBI)
    projected_operating_income_after_tax = (projected_operating_income*(1-projected_tax_rates))
    ### FCFF: EBI-Reinvestment
    projected_FCFF = projected_operating_income_after_tax - projected_reinvestment
    ### compute ROIC
    ROIC = (projected_operating_income_after_tax/invested_capital)
    ### Compute Terminal Value
    terminal_cost_of_capital = projected_cost_of_capital[-1:].values
    terminal_cost_of_equity = projected_cost_of_equity[-1:].values
    if terminal_growth_rate < 0:
        terminal_reinvestment_rate=0
    else:
        terminal_reinvestment_rate = terminal_growth_rate/(terminal_cost_of_capital+additional_return_on_cost_of_capital_in_perpetuity)
    terminal_revenue = projected_revneues[-1:].values * (1+terminal_growth_rate)
    terminal_operating_income = terminal_revenue * terminal_operating_margin
    terminal_operating_income_after_tax = terminal_operating_income*(1-marginal_tax_rate)
    terminal_reinvestment = terminal_operating_income_after_tax* terminal_reinvestment_rate
    terminal_FCFF = terminal_operating_income_after_tax - terminal_reinvestment
    terminal_value = terminal_FCFF/(terminal_cost_of_capital-terminal_growth_rate)
    termimal_discount_rate = (terminal_cost_of_capital-terminal_growth_rate)*(1+projected_cost_of_capital).prod()
    termimal_equity_discount_rate = (terminal_cost_of_equity-terminal_growth_rate)*(1+projected_cost_of_equity).prod()


    ### Concatinate Projected Values with termianl values
    projected_cost_of_capital_cumulative_with_terminal_rate =pd.concat([projected_cost_of_capital_cumulative,
                                                    pd.Series(termimal_discount_rate)])
    projected_cost_of_equity_cumulative_with_terminal_rate = pd.concat([projected_cost_of_equity_cumulative,
                                                    pd.Series(termimal_equity_discount_rate)])

    projected_revenue_growth = pd.concat([projected_revenue_growth,
                                        pd.Series(terminal_growth_rate)])
    projected_revneues =pd.concat([projected_revneues,
                                  pd.Series(terminal_revenue)])
    projected_tax_rates = pd.concat([projected_tax_rates,
                                   pd.Series(marginal_tax_rate)])
    projected_reinvestment = pd.concat([projected_reinvestment,
                                        pd.Series(terminal_reinvestment)])


    ### Estimate invested Capital
    invested_capital = pd.concat([invested_capital,
                                        pd.Series(np.NaN)])
    ### Estimate ROIC
    terminal_ROIC = terminal_cost_of_capital+additional_return_on_cost_of_capital_in_perpetuity
    ROIC = pd.concat([ROIC,
                      pd.Series(terminal_ROIC)])

    projected_operating_margins = pd.concat([projected_operating_margins,
                                        pd.Series(terminal_operating_margin)])
    projected_operating_income = pd.concat([projected_operating_income,
                                            pd.Series(terminal_operating_income)])
    projected_operating_income_after_tax = pd.concat([projected_operating_income_after_tax,
                                                      pd.Series(terminal_operating_income_after_tax)])
    projected_FCFF_value = pd.concat([projected_FCFF,
                                      pd.Series(terminal_value)])

    projected_FCFF = pd.concat([projected_FCFF,
                                pd.Series(terminal_FCFF)])

    projected_beta = pd.concat([projected_beta,
                                pd.Series(terminal_beta)])

    sales_to_capital_ratios = pd.concat([sales_to_capital_ratios,
                                pd.Series([terminal_sales_to_capital_ratio])])

    ### Add terminal cost of debt to the terminal year
    projected_after_tax_cost_of_debt_with_terminal = pd.concat([projected_after_tax_cost_of_debt,
                                                                pd.Series(projected_after_tax_cost_of_debt[-1:].values)])

    reinvestmentRate = projected_reinvestment/projected_operating_income_after_tax

    df_valuation = pd.DataFrame({"cumWACC":projected_cost_of_capital_cumulative_with_terminal_rate,
                                 "cumCostOfEquity":projected_cost_of_equity_cumulative_with_terminal_rate,
                                'beta':projected_beta,
                                 'ERP':ERP,
                                 'projected_after_tax_cost_of_debt':projected_after_tax_cost_of_debt_with_terminal,
                                'revenueGrowth':projected_revenue_growth,
                                "revneues":projected_revneues,
                                 "margins":projected_operating_margins,
                                 'ebit':projected_operating_income,
                                 "sales_to_capital_ratio":sales_to_capital_ratios,
                                "taxRate":projected_tax_rates,
                               'afterTaxOperatingIncome':projected_operating_income_after_tax,
                               "reinvestment":projected_reinvestment,
                               "invested_capital":invested_capital,
                               "ROIC":ROIC,
                               'reinvestmentRate':reinvestmentRate,
                               'FCFF':projected_FCFF,
                               'projected_FCFF_value':projected_FCFF_value})
    #### Add reinvestment rate and expected growth rate
    df_valuation['PVFCFF'] = df_valuation['FCFF']/df_valuation['cumWACC']
    value_of_operating_assets = df_valuation['PVFCFF'].sum()
    firm_value =  pd.Series(value_of_operating_assets + cash_and_non_operating_asset)[0]
    intrinsic_equity_present_value = firm_value - debt_value

    #### Future Frim, Debt and Equity Value
    cum_cost_of_debt_at_the_end_of_end_of_valuation =  (projected_after_tax_cost_of_debt+1).prod()
    cum_cost_of_capital_at_the_end_of_valuation =projected_cost_of_capital_cumulative[-1:].values[0]
    cum_cost_of_equity_at_the_end_of_valuation = projected_cost_of_equity_cumulative[-1:].values[0]

    ### FV Debt
    debt_future_value = cum_cost_of_debt_at_the_end_of_end_of_valuation * debt_value

    ### FV Frim
    firm_future_value = cum_cost_of_capital_at_the_end_of_valuation * cum_cost_of_capital_at_the_end_of_valuation

    ### FV Equity
    intrinsic_equity_future_value = intrinsic_equity_present_value * cum_cost_of_equity_at_the_end_of_valuation
    ## Returns
    def cum_return_calculator(value,
                            period,
                            append_nan=True):
        period = int(period)
        cum_return_series = pd.Series([1+value]*period).cumprod()
        if append_nan:
            cum_return_series = pd.concat([cum_return_series,pd.Series(np.NAN)])
        return(cum_return_series)

    acceptable_annualized_return_on_equity = ((cum_cost_of_equity_at_the_end_of_valuation)**(1/valuation_interval_in_years))-1
    expected_annualized_return_on_equity = ((intrinsic_equity_future_value/equity_value)**(1/valuation_interval_in_years))-1
    excess_annualized_return_on_equity = ((intrinsic_equity_present_value/equity_value)**(1/valuation_interval_in_years))-1
    total_excess_return_on_equity_during_valuation_cycle = (intrinsic_equity_present_value/equity_value)-1
    # print("acceptable_annualized_return_on_equity",acceptable_annualized_return_on_equity)
    # print("expected_annualized_return_on_equity",expected_annualized_return_on_equity)
    # print("excess_annualized_return_on_equity",excess_annualized_return_on_equity)
    # print("total_excess_return_on_equity_during_valuation_cycle",total_excess_return_on_equity_during_valuation_cycle)
    # print(pd.Series([1+acceptable_annualized_return_on_equity]*valuation_interval_in_years))
    cum_acceptable_annualized_return_on_equity = cum_return_calculator(value = acceptable_annualized_return_on_equity,
                                                                       period = valuation_interval_in_years,
                                                                       append_nan=True)
    cum_expected_annualized_return_on_equity = cum_return_calculator(value = expected_annualized_return_on_equity,
                                                                       period = valuation_interval_in_years,
                                                                       append_nan=True)
    cum_excess_annualized_return_on_equity = cum_return_calculator(value = excess_annualized_return_on_equity,
                                                                       period = valuation_interval_in_years,
                                                                       append_nan=True)
    df_valuation['cum_acceptable_annualized_return_on_equity'] = cum_acceptable_annualized_return_on_equity
    df_valuation['cum_expected_annualized_return_on_equity'] = cum_expected_annualized_return_on_equity
    df_valuation['cum_excess_annualized_return_on_equity'] = cum_excess_annualized_return_on_equity
    df_valuation['cum_excess_annualized_return_on_equity_realized'] = df_valuation['cum_expected_annualized_return_on_equity']/df_valuation['cumCostOfEquity']
    df_valuation['excess_annualized_return_on_equity'] = pd.concat([pd.Series([excess_annualized_return_on_equity]*int(valuation_interval_in_years)),pd.Series(np.NAN)])
    # print("valuation complete")
    return({'valuation':df_valuation,
            'firm_value':firm_value,
            'equity_value':intrinsic_equity_present_value,
            'cash_and_non_operating_asset':cash_and_non_operating_asset,
            'debt_value':debt_value,
            'value_of_operating_assets':value_of_operating_assets})


# In[ ]:


def point_estimate_describer(base_case_valuation):
    print('value of operating assets',np.round(base_case_valuation['value_of_operating_assets'],2),'\n',
        'cash and non operating asset',np.round(base_case_valuation['cash_and_non_operating_asset'],2),'\n',
        'debt value',np.round(base_case_valuation['debt_value'],2),'\n',
        'firm value',np.round(base_case_valuation['firm_value'],2),'\n',
        'Intrinsic Equity value',"${:.2f}".format(np.round(base_case_valuation['equity_value'],2)))
    df_valuation = base_case_valuation['valuation']
    df_valuation=  df_valuation.reset_index(drop=True)
    df_valuation['Year']= df_valuation.reset_index()['index']+1
    df_valuation.loc[df_valuation['Year'] == df_valuation['Year'].max(),"Year"] = 'Terminal'
    df_valuation= df_valuation.set_index("Year")
    return(df_valuation)


# ## Monte Carlo Simulation

# In[ ]:


def monte_carlo_valuator_multi_phase(
    risk_free_rate,
    ERP,
    equity_value,
    debt_value,
    unlevered_beta,
    terminal_unlevered_beta,
    year_beta_begins_to_converge_to_terminal_beta,
    current_pretax_cost_of_debt,
    terminal_pretax_cost_of_debt,
    year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt,
    current_effective_tax_rate,
    marginal_tax_rate,
    year_effective_tax_rate_begin_to_converge_marginal_tax_rate,
     revenue_base,
     revenue_growth_rate_cycle1_begin,
     revenue_growth_rate_cycle1_end,
     revenue_growth_rate_cycle2_begin,
     revenue_growth_rate_cycle2_end,
     revenue_growth_rate_cycle3_begin,
     revenue_growth_rate_cycle3_end,
    revenue_convergance_periods_cycle1,
    revenue_convergance_periods_cycle2,
    revenue_convergance_periods_cycle3,
    length_of_cylcle1,
    length_of_cylcle2,
    length_of_cylcle3,
    current_sales_to_capital_ratio,
    terminal_sales_to_capital_ratio,
    year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital,
    current_operating_margin,
    terminal_operating_margin,
    year_operating_margin_begins_to_converge_to_terminal_operating_margin,
    additional_return_on_cost_of_capital_in_perpetuity,
    cash_and_non_operating_asset,
    asset_liquidation_during_negative_growth,
    current_invested_capital,
    sample_size=1000,
    list_of_correlation_between_variables=[['additional_return_on_cost_of_capital_in_perpetuity','terminal_sales_to_capital_ratio',0.4],
                                           ['additional_return_on_cost_of_capital_in_perpetuity','terminal_operating_margin',.6]]):
    variables_distributsion = [risk_free_rate,
                                   ERP,
                                   equity_value,
                                   debt_value,
                                   unlevered_beta,
                                    terminal_unlevered_beta,
                                    year_beta_begins_to_converge_to_terminal_beta,
                                    current_pretax_cost_of_debt,
                                    terminal_pretax_cost_of_debt,
                                    year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt,
                                    current_effective_tax_rate,
                                    marginal_tax_rate,
                                    year_effective_tax_rate_begin_to_converge_marginal_tax_rate,
                                    revenue_base,
                                    revenue_growth_rate_cycle1_begin,
                                    revenue_growth_rate_cycle1_end,
                                    revenue_growth_rate_cycle2_begin,
                                    revenue_growth_rate_cycle2_end,
                                    revenue_growth_rate_cycle3_begin,
                                    revenue_growth_rate_cycle3_end,
                                    revenue_convergance_periods_cycle1,
                                    revenue_convergance_periods_cycle2,
                                    revenue_convergance_periods_cycle3,
                                    length_of_cylcle1,
                                    length_of_cylcle2,
                                    length_of_cylcle3,
                                    current_sales_to_capital_ratio,
                                    terminal_sales_to_capital_ratio,
                                    year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital,
                                    current_operating_margin,
                                    terminal_operating_margin,
                                    year_operating_margin_begins_to_converge_to_terminal_operating_margin,
                                    additional_return_on_cost_of_capital_in_perpetuity,
                                    cash_and_non_operating_asset,
                                    asset_liquidation_during_negative_growth,
                                    current_invested_capital]
    variable_names = ['risk_free_rate',
                                   'ERP',
                                   'equity_value',
                                   'debt_value',
                                   'unlevered_beta',
                                    'terminal_unlevered_beta',
                                    'year_beta_begins_to_converge_to_terminal_beta',
                                    'current_pretax_cost_of_debt',
                                    'terminal_pretax_cost_of_debt',
                                    'year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt',
                                    'current_effective_tax_rate',
                                    'marginal_tax_rate',
                                    'year_effective_tax_rate_begin_to_converge_marginal_tax_rate',
                                    'revenue_base',
                                    'revenue_growth_rate_cycle1_begin',
                                    'revenue_growth_rate_cycle1_end',
                                    'revenue_growth_rate_cycle2_begin',
                                    'revenue_growth_rate_cycle2_end',
                                    'revenue_growth_rate_cycle3_begin',
                                    'revenue_growth_rate_cycle3_end',
                                    'revenue_convergance_periods_cycle1',
                                    'revenue_convergance_periods_cycle2',
                                    'revenue_convergance_periods_cycle3',
                                    'length_of_cylcle1',
                                    'length_of_cylcle2',
                                    'length_of_cylcle3',
                                    'current_sales_to_capital_ratio',
                                    'terminal_sales_to_capital_ratio',
                                    'year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital',
                                    'current_operating_margin',
                                    'terminal_operating_margin',
                                    'year_operating_margin_begins_to_converge_to_terminal_operating_margin',
                                    'additional_return_on_cost_of_capital_in_perpetuity',
                                    'cash_and_non_operating_asset',
                                    'asset_liquidation_during_negative_growth',
                                    'current_invested_capital']
    ### The following variable  should have "year" in their definition but I did not think of ut. So I am adding them to list_of_columns_with_year_to_be_int
    list_of_columns_with_year_to_be_int = [s for s in variable_names if "year" in s] +['length_of_cylcle1','length_of_cylcle2','length_of_cylcle3',
                                                                                       'revenue_convergance_periods_cycle1',
                                                                                       'revenue_convergance_periods_cycle2','revenue_convergance_periods_cycle3']
    ### Build a DataFarame to have index - location of each variable in the correlation matrix
    dict_of_varible = dict(zip(variable_names,
                            range(0,len(variable_names))))
    df_variables = pd.DataFrame([dict_of_varible])

    ### Initaile Correlation Matrix
    R = ot.CorrelationMatrix(len(variables_distributsion))
    ### pair correlation between each variable
    for pair_of_variable in list_of_correlation_between_variables:
        location = df_variables[pair_of_variable[:2]].values[0]
        #print(location)
        R[int(location[0]),int(location[1])] = pair_of_variable[2]

    ### Build the correlation into composed distribution function
    ### For ot.NormalCopula The correlation matrix must be definite positive
    ### Here is an implementaion on how to get the nearest psd matirx https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
    copula = ot.NormalCopula(R)
    BuiltComposedDistribution = ot.ComposedDistribution(variables_distributsion,
                                                        copula)
    ### Generate samples
    generated_sample = BuiltComposedDistribution.getSample(sample_size)
    df_generated_sample = pd.DataFrame.from_records(generated_sample, columns= variable_names)
    df_generated_sample[list_of_columns_with_year_to_be_int] = df_generated_sample[list_of_columns_with_year_to_be_int].apply(lambda x: round(x))
    print("Scenario Generation Complete", df_generated_sample.shape)
    df_generated_sample['full_valuation']= df_generated_sample.apply(lambda row:
                                                                    valuator_multi_phase(
                                                                        risk_free_rate = row['risk_free_rate'],
                                                                        ERP = row['ERP'],
                                                                        equity_value = row['equity_value'],
                                                                        debt_value = row['debt_value'],
                                                                        unlevered_beta = row['unlevered_beta'],
                                                                        terminal_unlevered_beta = row['terminal_unlevered_beta'],
                                                                        year_beta_begins_to_converge_to_terminal_beta = row['year_beta_begins_to_converge_to_terminal_beta'],
                                                                        current_pretax_cost_of_debt = row['current_pretax_cost_of_debt'],
                                                                        terminal_pretax_cost_of_debt = row['terminal_pretax_cost_of_debt'],
                                                                        year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt = row['year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt'],
                                                                        current_effective_tax_rate = row['current_effective_tax_rate'],
                                                                        marginal_tax_rate = row['marginal_tax_rate'],
                                                                        year_effective_tax_rate_begin_to_converge_marginal_tax_rate = row['year_effective_tax_rate_begin_to_converge_marginal_tax_rate'],
                                                                        revenue_base = row['revenue_base'],
                                                                        revenue_growth_rate_cycle1_begin = row['revenue_growth_rate_cycle1_begin'],
                                                                        revenue_growth_rate_cycle1_end = row['revenue_growth_rate_cycle1_end'],
                                                                        revenue_growth_rate_cycle2_begin = row['revenue_growth_rate_cycle2_begin'],
                                                                        revenue_growth_rate_cycle2_end = row['revenue_growth_rate_cycle2_end'],
                                                                        revenue_growth_rate_cycle3_begin = row['revenue_growth_rate_cycle3_begin'],
                                                                        revenue_growth_rate_cycle3_end = row['revenue_growth_rate_cycle3_end'],
                                                                        revenue_convergance_periods_cycle1 = row['revenue_convergance_periods_cycle1'],
                                                                        revenue_convergance_periods_cycle2 = row['revenue_convergance_periods_cycle2'],
                                                                        revenue_convergance_periods_cycle3 = row['revenue_convergance_periods_cycle3'],
                                                                        length_of_cylcle1 = row['length_of_cylcle1'],
                                                                        length_of_cylcle2 = row['length_of_cylcle2'],
                                                                        length_of_cylcle3 = row['length_of_cylcle3'],
                                                                        current_sales_to_capital_ratio = row['current_sales_to_capital_ratio'],
                                                                        terminal_sales_to_capital_ratio = row['terminal_sales_to_capital_ratio'],
                                                                        year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital = row['year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital'],
                                                                        current_operating_margin = row['current_operating_margin'],
                                                                        terminal_operating_margin = row['terminal_operating_margin'],
                                                                        year_operating_margin_begins_to_converge_to_terminal_operating_margin = row['year_operating_margin_begins_to_converge_to_terminal_operating_margin'],
                                                                        additional_return_on_cost_of_capital_in_perpetuity = row['additional_return_on_cost_of_capital_in_perpetuity'],
                                                                        cash_and_non_operating_asset = row['cash_and_non_operating_asset'],
                                                                        asset_liquidation_during_negative_growth=row['asset_liquidation_during_negative_growth'],
                                                                        current_invested_capital=row['current_invested_capital']),
                                                                        axis=1)
    ### extract the valuation result
    df_generated_sample['valuation'] = df_generated_sample['full_valuation'].apply(lambda x: x['valuation'])
    df_generated_sample['equity_valuation'] = df_generated_sample['full_valuation'].apply(lambda x: x['equity_value'])
    df_generated_sample['firm_valuation'] = df_generated_sample['full_valuation'].apply(lambda x: x['firm_value'])
    df_generated_sample['terminal_revenue'] = df_generated_sample['valuation'].apply(lambda x: x['revneues'].values[-1])
    df_generated_sample['terminal_operating_margin'] = df_generated_sample['valuation'].apply(lambda x: x['margins'].values[-1])
    df_generated_sample['terminal_reinvestmentRate'] = df_generated_sample['valuation'].apply(lambda x: x['reinvestmentRate'].values[-1])
    df_generated_sample['terminal_afterTaxOperatingIncome'] = df_generated_sample['valuation'].apply(lambda x: x['afterTaxOperatingIncome'].values[-1])
    df_generated_sample['terminal_FCFF'] = df_generated_sample['valuation'].apply(lambda x: x['FCFF'].values[-1])
    df_generated_sample['cumWACC'] = df_generated_sample['valuation'].apply(lambda x: x['cumWACC'][:-1].values)
    df_generated_sample['cumCostOfEquity'] = df_generated_sample['valuation'].apply(lambda x: x['cumCostOfEquity'][:-1].values)
    df_generated_sample['cum_acceptable_annualized_return_on_equity'] = df_generated_sample['valuation'].apply(lambda x: x['cum_acceptable_annualized_return_on_equity'][:-1].values)
    df_generated_sample['cum_expected_annualized_return_on_equity'] = df_generated_sample['valuation'].apply(lambda x: x['cum_expected_annualized_return_on_equity'][:-1].values)
    df_generated_sample['cum_excess_annualized_return_on_equity'] = df_generated_sample['valuation'].apply(lambda x: x['cum_excess_annualized_return_on_equity'][:-1].values)
    df_generated_sample['cum_excess_annualized_return_on_equity_realized'] = df_generated_sample['valuation'].apply(lambda x: x['cum_excess_annualized_return_on_equity_realized'][:-1].values)
    df_generated_sample['excess_annualized_return_on_equity'] = df_generated_sample['valuation'].apply(lambda x: x['excess_annualized_return_on_equity'][:-1].values)
    df_generated_sample['ROIC'] = df_generated_sample['valuation'].apply(lambda x: x['ROIC'][:-1].values)
    df_generated_sample['invested_capital'] = df_generated_sample['valuation'].apply(lambda x: x['invested_capital'][:-1].values)
    return(df_generated_sample)


# ### Plotly Charts

# In[ ]:


def histogram_plotter_plotly(data,
                              colmn_name,
                              xlabel,
                              title='Data',
                              bins=30,
                              percentile=[15,50,85],
                              color=['green','yellow','red'],
                              histnorm='percent',
                              marginal=None,
                              height=470,
                              width=670):
    """Plot Historgam via Plotly"""
    fig = px.histogram(data,
                       x=colmn_name,
                       histnorm=histnorm,
                       nbins=bins,
                       labels={colmn_name:xlabel},
                       marginal=marginal)
    ### Make an educated guess on the y_max for line on the historgram
    n, bin_edges = np.histogram(data[colmn_name],bins=bins,density=False)
    bin_probability = n/float(n.sum())
    y_max = np.max(n/(n.sum())*100) *1.65
    ### Ad trace of percentiles
    for i in range(len(percentile)):
        fig = fig.add_trace(go.Scatter(x=[np.percentile(data[colmn_name],percentile[i]), np.percentile(data[colmn_name],percentile[i])],
                                       y=(0,y_max),
                                       mode="lines",
                                       name= str(percentile[i])+' Percentile',
                                       marker=dict(color=color[i])))
        #fig = fig.add_vline(x = np.percentile(data[colmn_name],percentile[i]), line_dash = 'dash',line_color=color[i])
        #print(str(percentile[i])+" Percentile",np.percentile(data[colmn_name],percentile[i]))
        fig.update_layout(height=height, width=width,title=title,
                          legend=dict(orientation="v"))
    return(fig)

def ecdf_plotter_plotly(data,
                              colmn_name,
                              xlabel,
                              title='Data',
                              percentile=[15,50,85],
                              color=['green','yellow','red'],
                              marginal=None,
                              height=500,
                              width=700):
    """Plot ECDF via Plotly"""
    fig = px.ecdf(data,
                     x=colmn_name,
                     labels={colmn_name:xlabel},
                     marginal=marginal)
    for i in range(len(percentile)):
        fig = fig.add_trace(go.Scatter(x=[np.percentile(data[colmn_name],percentile[i]), np.percentile(data[colmn_name],percentile[i])],
                                       y=(0,1),
                                       mode="lines",
                                       name= str(percentile[i])+' Percentile',
                                       marker=dict(color=color[i])))
        #fig = fig.add_vline(x = np.percentile(data[colmn_name],percentile[i]), line_dash = 'dash',line_color=color[i])
        #print(str(percentile[i])+" Percentile",np.percentile(data[colmn_name],percentile[i]))
        fig.update_layout(height=height, width=width,title=title,
                          legend=dict(orientation="v"))
    return(fig)


# In[ ]:


def time_series_plotly(df_data,
                       x,
                       yleft,
                       yright,
                       height=500,
                       width=1600,
                       title=None):
    """ Graph 2 time series on 2 different y-axis"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=df_data[x], y=df_data[yleft], name=yleft),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=df_data[x], y=df_data[yright], name=yright),
        secondary_y=True,
    )
    fig = fig.update_layout(height=height, width=width,title=title)
    return(fig)

def plotly_line_bar_chart(df_data,
                       x,
                       ybar,
                       yline,
                       height=500,
                       width=1600,
                       rangemode=None,
                       title=None):
    """ Graph 2 time series on 2 different y-axis"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    #fig.update_yaxes(rangemode='tozero')

    for bar_var in ybar:
        # Add traces
        fig.add_trace(
            go.Bar(x=df_data[x], y=df_data[bar_var],name=bar_var),
            secondary_y=False
            )
    #fig.update_yaxes(rangemode='tozero')
    for line_var in yline:
        fig.add_trace(
            go.Scatter(x=df_data[x], y=df_data[line_var],name=line_var),
            secondary_y=True,
            )
    if rangemode != None:
        fig.update_yaxes(rangemode=rangemode)
    fig = fig.update_layout(height=height, width=width,title=title)
    return(fig)


def plotly_line_dash_bar_chart(df_data,
                       x,
                       ybar,
                       yline,
                       ydash,
                       height=500,
                       width=1600,
                       rangemode=None,
                       title=None,
                       barmode='group',
                       texttemplate= "%{value}"
                       ):
    """ Graph 2 time series on 2 different y-axis"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    #fig.update_yaxes(rangemode='tozero')

    for bar_var in ybar:
        # Add traces
        fig.add_trace(
            go.Bar(x=df_data[x],
                   y=df_data[bar_var],
                   name=bar_var,
                   text = df_data[bar_var],
                   textposition="inside",
                   texttemplate= texttemplate,
                   textfont_color="white"),
            secondary_y=False,
            )
    for line_var in yline:
        fig.add_trace(
            go.Scatter(x=df_data[x],
                       y=df_data[line_var],
                       name=line_var
                       ),
            secondary_y=True,
            )

    for dash_var in ydash:
        fig.add_trace(
            go.Scatter(x=df_data[x],
                       y=df_data[dash_var],
                       name=dash_var,
                       line = dict(dash='dot')),
            secondary_y=True,
            )
    if rangemode != None:
        fig.update_yaxes(rangemode=rangemode)
    fig = fig.update_layout(height=height,
                            width=width,
                            title=title,
                            barmode=barmode)
    return(fig)


# In[ ]:


def line_plotter_with_error_bound(df_data,
                                  x,
                                  list_of_mid_point,
                                  list_of_lower_bound,
                                  list_of_upper_bound,
                                  list_of_bar=[],
                                  list_of_name=[],
                                  list_of_fillcolor= ['rgba(68, 68, 68, 0.3)'],
                                  list_of_line_color= ['rgb(31, 119, 180)'],
                                  title=None,
                                  yaxis_title=None,
                                  height=600,
                                  width=900):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    #fig.update_yaxes(rangemode='tozero')
    for mid_point,lower_bound,upper_bound,name,fillcolor,line_color in zip(list_of_mid_point,
                                                                            list_of_lower_bound,
                                                                            list_of_upper_bound,
                                                                            list_of_name,
                                                                            list_of_fillcolor,
                                                                            list_of_line_color):
        # Add traces
        fig.add_trace (
            go.Scatter(
                name = name,
                x=df_data[x],
                y=df_data[mid_point],
                mode='lines',
                line=dict(color=line_color)
                ))
        fig.add_trace(
            go.Scatter(
                name= name + ' UB',
                x = df_data[x],
                y = df_data[upper_bound],
                mode='lines',
                marker=dict(color=line_color),
                line=dict(width=0),
                showlegend=False
                ))
        fig.add_trace(
            go.Scatter(
        name= name + ' LB',
        x= df_data[x],
        y=df_data[lower_bound],
        marker=dict(color=line_color),
        line=dict(width=0),
        mode='lines',
        fillcolor= fillcolor,
        fill='tonexty',
        showlegend=False)
        )

    for bar_var in list_of_bar:
        fig.add_trace(
            go.Bar(x=df_data[x],
                   y=df_data[bar_var],
                   name=bar_var,
                   text = df_data[bar_var],
                   marker=dict(color='rgba(46, 120, 237,.6)'),
                   textposition="inside",
                   #texttemplate= texttemplate,
                   textfont_color="white"),
            secondary_y=True,
            )

    fig.update_layout(
        yaxis2=dict(
        side="right",
        #rangemode='tozero',
        range=[-.05, .3],
        #overlaying="y",
        #tickmode="sync"
        )
        )
    # fig.update_yaxes(rangemode='tozero')
    fig.update_layout(yaxis_title= yaxis_title,
                      title= title,
                      hovermode="x",
                      height=height,
                      width=width,)
    return(fig)


# ### Valuation Describer

# In[ ]:


def return_values_from_list_extractor(df_data,
                                        col,
                                        add_1=True):
    "This function is to get the cost stats of specied col by year"
    df_cost_of_cap = pd.DataFrame(list(df_data[col].values))
    if add_1:
        df_cost_of_cap[-1] = 1.00
    else:
        df_cost_of_cap[-1] = 0
    df_cost_of_cap = pd.melt(df_cost_of_cap,
                                var_name=['year'],
                                value_name=col).dropna()
    df_cost_of_cap = df_cost_of_cap.groupby(['year'])[[col]].describe().reset_index()
    df_cost_of_cap = data_frame_flattener(df_cost_of_cap)
    df_cost_of_cap = df_cost_of_cap.set_index("year")
    return(df_cost_of_cap)

def return_values_from_list_extractor_step2(df_data,
                                            cum_ret_col,
                                            ret_col):
    list_of_df_return = []
    for col in cum_ret_col:
        df_res = return_values_from_list_extractor(df_data,
                                                   col=col)
        list_of_df_return.append(df_res)
    for col in ret_col:
        df_res = return_values_from_list_extractor(df_data,
                                                   col=col,
                                                   add_1=False)
        list_of_df_return.append(df_res)
    df_returns = pd.concat(list_of_df_return,axis=1)
    df_returns = df_returns.reset_index(drop=True).reset_index().rename(columns={"index":"year"})
    return(df_returns)


# In[ ]:


def valuation_describer(df_intc_valuation,
                        sharesOutstanding=1):
    """Describe stats of monte dcf carlo simulation"""
    ### Get the Equity value at eahc percentile
    current_market_cap = df_intc_valuation['equity_value'].median()
    percentiles=np.arange(0, 110, 10)
    equity_value_at_each_percentile = np.percentile(df_intc_valuation['equity_valuation'],
                                                    percentiles)
    equity_value_at_20_percentile=equity_value_at_each_percentile[2]
    equity_value_at_80_percentile=equity_value_at_each_percentile[8]
    df_valuation_res = pd.DataFrame({"percentiles":percentiles,
                                     "equity_value":equity_value_at_each_percentile})
    df_valuation_res['current_market_cap'] = current_market_cap
    df_valuation_res['current_price_per_share'] = current_market_cap/sharesOutstanding
    df_valuation_res['equity_value_per_share'] = df_valuation_res['equity_value']/sharesOutstanding
    df_valuation_res['Price/Value']= df_valuation_res['current_market_cap']/df_valuation_res['equity_value']
    df_valuation_res['PNL']= (df_valuation_res['equity_value']/df_valuation_res['current_market_cap'])-1
    ### Histogram
    fig = histogram_plotter_plotly(data=df_intc_valuation,
                              colmn_name ='equity_valuation',
                              xlabel ='Market Cap',
                              title='Intrinsic Equity Value Distribution',
                              bins=200,
                              percentile=[15,50,85],
                              color=['green','yellow','red'],
                              histnorm='percent',
                              height=510,
                              width=720)
    fig = fig.add_vline(x = current_market_cap, line_dash = 'dash',line_color='black',
                        annotation_text="-Current Market Cap",
                        annotation_font_size=10)
    ### Plot cummultaive distribution of intrincsict equity value
    fig_cdf = ecdf_plotter_plotly(data=df_intc_valuation,
                              colmn_name ='equity_valuation',
                              xlabel ='Market Cap',
                              title='Intrinsic Equity Value Cumulative Distribution',
                              percentile=[15,50,85],
                              color=['green','yellow','red'],
                              marginal='histogram',
                              height=510,
                              width=720)
    fig_cdf = fig_cdf.add_vline(x = current_market_cap, line_dash = 'dash',line_color='black',annotation_text="-Current Market Cap", annotation_font_size=10)
    ### Model Correlation Chart
    fig_model_correlation_chart = px.bar(df_intc_valuation.rename(columns=dict(zip(df_intc_valuation.columns,
                                     [c.replace("_"," ") for c in df_intc_valuation.columns]))).corr(method='pearson')[['equity valuation']].sort_values(
                                         "equity valuation",ascending=False).reset_index(),
       x='index',
       y='equity valuation',
       title='Model Variable Pearson Correlation with Equity Intrinsic Value',
        height=730,
       width=1600,
       text_auto='.2f',
       labels={'index':'Model Variable',
               'equity valuation':'Correlation'})
    #### Return on Investment
    df_returns = return_values_from_list_extractor_step2(df_intc_valuation,
                                            cum_ret_col = ['cumWACC', 'cumCostOfEquity','cum_acceptable_annualized_return_on_equity','cum_expected_annualized_return_on_equity',
                                                           'cum_excess_annualized_return_on_equity','cum_excess_annualized_return_on_equity_realized'],
                                            ret_col=['excess_annualized_return_on_equity'])

    ### Plot returns
    fig_return = line_plotter_with_error_bound(df_data = df_returns,
                                  x='year',
                                  list_of_mid_point = ['cum_expected_annualized_return_on_equity 50%','cumCostOfEquity 50%'],
                                  list_of_lower_bound = ['cum_expected_annualized_return_on_equity min','cumCostOfEquity min'],
                                  list_of_upper_bound = ['cum_expected_annualized_return_on_equity max','cumCostOfEquity max'],
                                list_of_bar=['excess_annualized_return_on_equity 50%'],
                                  list_of_name=['Cum Expected Return','Cost of Equity'],
                                  list_of_fillcolor= ['rgba(59, 237, 157, 0.5)','rgba(255,84,167, 0.3)'],
                                  list_of_line_color= ['rgb(37, 162, 111)','rgb(238, 72, 103)'],
                                  title='Return on Equity Investment',
                                  yaxis_title='Cum Return',
                                  height=500,
                                  width=1400)
    #### ROIC and Invested Capital
    df_roic = return_values_from_list_extractor_step2(df_data = df_intc_valuation,
                                            cum_ret_col = [],
                                            ret_col = ['ROIC','invested_capital'])[1:]

    fig_roic_inv = plotly_line_dash_bar_chart(df_roic,
                       x='year',
                       ybar=['invested_capital 50%'],
                       yline=['ROIC 50%'],
                       ydash=[],
                       height=500,
                       width=1600,
                       rangemode=None,
                       title='Simulated Median ROIC and Invested Capital',
                       barmode='group',
                       #texttemplate= "%{value}"
                       ).update_layout(hovermode="x")


    fig_model_correlation_chart.show()
    fig_roic_inv.show()
    for col in ['revenue_growth_rate_cycle1_begin',
                'revenue_growth_rate_cycle1_end',
                'revenue_growth_rate_cycle2_begin',
                'revenue_growth_rate_cycle2_end',
                'revenue_growth_rate_cycle3_begin',
                'revenue_growth_rate_cycle3_end'
                'revenue_growth_rate',
                'risk_free_rate','ERP',
                'additional_return_on_cost_of_capital_in_perpetuity',
                'terminal_reinvestmentRate',
                'terminal_afterTaxOperatingIncome',
                'firm_valuation',
                'terminal_FCFF',
                'unlevered_beta',
                'terminal_unlevered_beta',
                'current_sales_to_capital_ratio',
                'terminal_sales_to_capital_ratio',
                'current_operating_margin',
                'terminal_operating_margin',
                'terminal_revenue']:
                try:
                    histogram_plotter_plotly(df_intc_valuation,
                                             colmn_name=col,
                                             xlabel=col.replace("_"," "),
                                             bins=200).show()
                except:
                    pass
    fig_cdf.show()
    fig.show()
    fig_return.show()
    return(df_valuation_res)


def get_sector_yfinance(ticker_symbol):
    """Fetch the sector for a given ticker using yfinance."""
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info
    return info.get('sector', None)

def get_industry_yfinance(ticker_symbol):
    """Fetch the sector for a given ticker using yfinance."""
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info
    return info.get('sector', None)

def get_stocks_from_same_industry(ticker_symbol):
    """Fetch stocks from the same industry as the provided ticker."""
    # Get the sector for the given ticker using yfinance
    sector = get_industry_yfinance(ticker_symbol)

    if not sector:
        print(f"Could not find industry for {ticker_symbol}")
        return None

    # Initialize the screener from yahooquery
    s = Screener()

    # Using sector to screen stocks
    screen_key = f"ms_{sector.lower()}"
    
    if screen_key not in s.available_screeners:
        print(f"No predefined screener available for sector: {sector}")
        return None

    data = s.get_screeners(screen_key)

    # Convert data to DataFrame for easier handling
    df = pd.DataFrame(data[screen_key]['quotes'])
    
    return df

def calculate_rolling_beta(stock_data, market_data, window_size):
    stock_returns = stock_data['Adj Close'].pct_change().dropna()
    market_returns = market_data['Adj Close'].pct_change().dropna()

    rolling_cov = stock_returns.rolling(window=window_size).cov(market_returns)
    rolling_var = market_returns.rolling(window=window_size).var()

    rolling_beta = rolling_cov / rolling_var
    return rolling_beta.dropna()

def get_unlevered_beta(ticker):
    stock = yf.Ticker(ticker)

    # Get levered beta
    levered_beta = stock.info['beta']
    if not levered_beta:
        return None

    # Get debt and equity values
    market_cap = stock.info['marketCap'] / 10**9
    long_term_debt = stock.balance_sheet.loc["Long Term Debt"][0] / 10**9 if "Long Term Debt" in stock.balance_sheet.index else 0
    short_term_debt = stock.balance_sheet.loc["Short Term Debt"][0] / 10**9 if "Short Term Debt" in stock.balance_sheet.index else 0
    debt_value = long_term_debt + short_term_debt
    equity_value = market_cap

    # Calculate the effective tax rate
    income_statement = stock.financials
    pretax_income = float(income_statement.loc["Pretax Income"].iloc[0])
    income_tax_expense = float(income_statement.loc["Tax Provision"].iloc[0])
    effective_tax_rate = income_tax_expense / pretax_income
    T = effective_tax_rate

    # Calculate unlevered beta
    return levered_beta / (1 + ((1 - T) * (debt_value / equity_value)))


def get_pretax_cost_of_debt(ticker):
    """Compute the pre-tax cost of debt for a given ticker."""
    stock = yf.Ticker(ticker)

    income_statement = stock.financials
    balance_sheet = stock.balance_sheet

    # Interest Expense from the income statement
    interest_expense = float(income_statement.loc["Interest Expense"].iloc[0]) if "Interest Expense" in income_statement.index else 0

    # Average Total Debt calculation
    current_long_term_debt = float(balance_sheet.loc["Long Term Debt"].iloc[0]) if "Long Term Debt" in balance_sheet.index else 0
    previous_long_term_debt = float(balance_sheet.loc["Long Term Debt"].iloc[1]) if "Long Term Debt" in balance_sheet.index else 0

    current_short_term_debt = float(balance_sheet.loc["Short Term Debt"].iloc[0]) if "Short Term Debt" in balance_sheet.index else 0
    previous_short_term_debt = float(balance_sheet.loc["Short Term Debt"].iloc[1]) if "Short Term Debt" in balance_sheet.index else 0

    average_debt = (current_long_term_debt + current_short_term_debt + previous_long_term_debt + previous_short_term_debt) / 2

    # Calculate the pre-tax cost of debt
    if average_debt == 0:
        return 0
    else:
        return interest_expense / average_debt

def get_year_cost_of_debt_converges(ticker, comparable_tickers):
  """Compute the year when the cost of debt converges to the industry average."""
  # Get the current pre-tax cost of debt for the given ticker
  current_pretax_cost_of_debt = get_pretax_cost_of_debt(ticker)
  if not current_pretax_cost_of_debt:
    return None # No cost of debt available
  
  # Get the pre-tax cost of debt for each comparable ticker
  pretax_costs_of_debt = [get_pretax_cost_of_debt(ticker) for ticker in comparable_tickers]
  pretax_costs_of_debt = [cost for cost in pretax_costs_of_debt if cost is not None] # Remove None values
  
  # Calculate the industry average pre-tax cost of debt
  industry_average_pretax_cost_of_debt = sum(pretax_costs_of_debt) / len(pretax_costs_of_debt)
  
  # Estimate the terminal pre-tax cost of debt using a weighted average
  omega = 0.5 # Weight given to the company's current pre-tax cost of debt
  terminal_pretax_cost_of_debt = omega * current_pretax_cost_of_debt + (1 - omega) * industry_average_pretax_cost_of_debt
  
  # Assume a linear convergence from the current to the terminal cost of debt
  # Use the equation y = mx + b, where y is the cost of debt, x is the year, m is the slope, and b is the intercept
  # Solve for x when y equals the terminal cost of debt
  slope = (terminal_pretax_cost_of_debt - current_pretax_cost_of_debt) / DURATION # DURATION is the number of years for valuation
  intercept = current_pretax_cost_of_debt
  year_cost_of_debt_converges = (terminal_pretax_cost_of_debt - intercept) / slope
  
  return year_cost_of_debt_converges

def get_marginal_tax_rate(ticker):
    """Compute the marginal tax rate for a given ticker using yfinance."""
    # Get the income statement from yfinance
    stock = yf.Ticker(ticker)
    income_statement = stock.financials

    # Get the income before tax and income tax expense from the income statement
    income_before_tax = float(income_statement.loc["Pretax Income"].iloc[0])
    income_tax_expense = float(income_statement.loc["Tax Provision"].iloc[0])

    # Calculate the marginal tax rate as the ratio of income tax expense to income before tax
    marginal_tax_rate = income_tax_expense / income_before_tax

    # Return the marginal tax rate as a percentage
    return marginal_tax_rate

def get_ttm_total_revenue(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    
    # Fetch the total revenue for the TTM
    ttm_revenue = ticker.info['totalRevenue']
    
    return ttm_revenue


def fetch_growth_estimate(ticker):
    # Fetch the analysis data
    analysis_data = si.get_analysts_info(ticker)

    print(analysis_data)

    # Extract growth estimates
    growth_estimates = analysis_data["Growth Estimates"]

    # Locate the 'Next 5 Years' growth estimate for the given ticker
    growth_next_5_years_row = growth_estimates[growth_estimates["Growth Estimates"] == "Next 5 Years (per annum)"]
    growth_next_5_years = growth_next_5_years_row[ticker].values[0]
    growth_next_1_year_row = growth_estimates[growth_estimates["Growth Estimates"] == "Next Year"]
    growth_next_1_year = growth_next_1_year_row[ticker].values[0]

    growth5 = float(growth_next_5_years.strip('%'))
    growth1 = float(growth_next_1_year.strip('%'))
    
    return growth1,growth5


def estimate_cycle_length(growth1, growth5):
    """
    Estimates the cycle length based on linear interpolation of the growth rate.
    Args:
    - growth1 (float): Initial growth rate at the beginning of the cycle (in %).
    - growth5 (float): Growth rate at the end of the cycle (in %).

    Returns:
    - int: Estimated number of years the cycle will last.
    """

    # Assuming linear change in growth rate
    growth_rate_change_per_year = (growth5 - growth1) / 4

    # If the growth rate doesn't change (i.e., change per year is 0), return 5 as the default cycle length
    if growth_rate_change_per_year == 0:
        return 5

    # Estimate the number of years required for the growth rate to reach the end rate
    years = (growth5 - growth1) / growth_rate_change_per_year

    return int(years)

def get_sales_to_capital_ratio(ticker):
    # Fetch financial data using yfinance
    company = yf.Ticker(ticker)
    
    # Get annual income statement and balance sheet
    income_statement = company.financials
    balance_sheet = company.balance_sheet

    # Extract sales (revenue) from the income statement
    sales = income_statement.loc["Total Revenue"][0]

    # Extract total debt and shareholder's equity from the balance sheet
    total_debt = balance_sheet.loc["Long Term Debt"][0] if "Long Term Debt" in balance_sheet.index else 0
    shareholders_equity = balance_sheet.loc["Total Equity Gross Minority Interest"][0]

    # Calculate sales-to-capital ratio
    sales_to_capital_ratio = sales / (total_debt + shareholders_equity)

    return sales_to_capital_ratio

def estimate_terminal_ratio_from_comparables(target_ticker, comparable_tickers):
    # List to store the sales-to-capital ratios for comparables
    ratios = []
    
    # Iterate through each comparable ticker and compute its sales-to-capital ratio
    for ticker in comparable_tickers:
        try:
            ratio = get_sales_to_capital_ratio(ticker)
            ratios.append(ratio)
        except:
            # Exception handling for tickers that might cause errors
            print(f"Could not fetch data for {ticker}")
            continue

    # Compute the median of the sales-to-capital ratios as the estimated terminal ratio
    terminal_ratio = np.median(ratios)
    
    return terminal_ratio

def years_to_converge(current_ratio, terminal_ratio, threshold_percentage=0.05):
    """
    Calculate the number of years required for the current ratio to converge towards the terminal ratio.

    :param current_ratio: Current sales to capital ratio.
    :param terminal_ratio: Terminal sales to capital ratio.
    :param threshold_percentage: Threshold percentage to consider as convergence.
    :return: Number of years to begin convergence.
    """

    # Check for potential division by zero
    if abs(terminal_ratio - current_ratio) < 1e-9:
        return 0  # The ratios are essentially the same

    # Calculate the yearly change (assuming linear convergence)
    yearly_change = threshold_percentage * (terminal_ratio - current_ratio)

    # Calculate the number of years required for convergence
    years = (terminal_ratio - current_ratio) / yearly_change

    # Return the absolute value of years, rounded up
    return float(years)

def get_current_operating_margin(ticker):
    # Fetch the company data
    company = yf.Ticker(ticker)
    
    # Get the annual income statement
    income_statement = company.financials

    # Extract operating income and total revenue
    operating_income = income_statement.loc["Operating Income"][0]
    total_revenue = income_statement.loc["Total Revenue"][0]

    # Calculate and return the operating margin
    return operating_income / total_revenue

def estimate_terminal_operating_margin(comparable_tickers):
    margins = []

    for ticker in comparable_tickers:
        try:
            margin = get_current_operating_margin(ticker)
            margins.append(margin)
        except:
            print(f"Couldn't fetch data for {ticker}. Skipping...")

    if not margins:
        raise ValueError("Could not fetch data for any comparables")

    # Return the average of the margins as the estimated terminal margin
    return sum(margins) / len(margins)

def year_margin_begins_to_converge(current_operating_margin, terminal_operating_margin, threshold=0.05):
    """
    Calculate the year when the current operating margin begins to converge to the terminal operating margin.
    
    Parameters:
    - current_operating_margin: Current operating margin of the company.
    - terminal_operating_margin: Estimated terminal operating margin based on industry comparables.
    - threshold: Convergence threshold. The year when the difference between the current and terminal margin
                 is less than this threshold will be returned.
    
    Returns:
    - Year when the current margin begins to converge to the terminal margin.
    """
    
    # If the current margin is already close to the terminal margin, convergence might not be needed.
    if abs(current_operating_margin - terminal_operating_margin) < threshold:
        return 0
    
    # Initialize year count
    year = 0
    
    # Loop until convergence is achieved
    while current_operating_margin - terminal_operating_margin > threshold:
        # Linearly converge the current margin to the terminal margin
        current_operating_margin = (current_operating_margin + terminal_operating_margin) / 2
        year += 1
        
        # Safety mechanism to prevent infinite loops
        if year > 100:  
            raise ValueError("Convergence taking too long. Check the values and threshold.")
    
    return year

def get_invested_capital(ticker):
    # Fetch financial data using yfinance
    company = yf.Ticker(ticker)
    
    # Get annual balance sheet
    balance_sheet = company.balance_sheet

    # Extract necessary data
    total_debt = balance_sheet.loc["Long Term Debt"][0] if "Long Term Debt" in balance_sheet.index else 0
    total_equity = balance_sheet.loc["Total Equity Gross Minority Interest"][0]
    cash = balance_sheet.loc["Cash"][0] if "Cash" in balance_sheet.index else 0
    cash_equivalents = balance_sheet.loc["Cash Equivalents"][0] if "Cash Equivalents" in balance_sheet.index else 0

    # Compute invested capital: total debt + total equity - cash - cash equivalents
    invested_capital = total_debt + total_equity - cash - cash_equivalents
    
    return invested_capital

# Define the URL for the API endpoint
TICKER = "AAPL"
ENDPOINT = "https://query1.finance.yahoo.com/v7/finance/download/{}"
TICKER_SP500 = "^GSPC"
DURATION = 5
TODAY = int(datetime.now().timestamp())
TEN_YEARS_AGO = int((datetime.now() - pd.DateOffset(years=DURATION)).timestamp())
urlRFR = "https://query1.finance.yahoo.com/v7/finance/download/%5ETNX?period1=0&period2=9999999999&interval=1d&events=history&includeAdjustedClose=true"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

responseRFR = requests.get(urlRFR, headers=headers)

if responseRFR.status_code == 200:
    content = responseRFR.text
    lines = content.strip().split("\n")
    
    if len(lines) < 2:  # Check if there's at least a header and one data line
        print("Error: Not enough data lines in the response.")
        exit()
    
    last_line = lines[-1]
    values = last_line.split(",")
    
    if len(values) < 4:  # Check if there are enough values in the last line
        print("Error: Not enough values in the data line.")
        exit()

    RFR = float(values[3])
    print(f"The 10-year treasury yield in USA is {RFR}%")
else:
    print(f"Error: The request failed with status code {responseRFR.status_code}. Response: {responseRFR.text}")


# Fetch S&P 500 historical data
urlSP500 = ENDPOINT.format(TICKER_SP500) + f"?period1={TEN_YEARS_AGO}&period2={TODAY}&interval=1d&events=history&includeAdjustedClose=true"
responseSP500 = requests.get(urlSP500, headers=headers)
if responseSP500.status_code != 200:
    raise Exception("Error fetching S&P 500 data.")
dataSP500 = pd.read_csv(urlSP500, parse_dates=['Date'], index_col='Date')

urlCOMPANY = ENDPOINT.format(TICKER) + f"?period1={TEN_YEARS_AGO}&period2={TODAY}&interval=1d&events=history&includeAdjustedClose=true"
responseCOMPANY = requests.get(urlCOMPANY, headers=headers)
if responseCOMPANY.status_code != 200:
    raise Exception("Error fetching company data.")
dataCOMPANY = pd.read_csv(io.StringIO(responseCOMPANY.text), parse_dates=['Date'], index_col='Date')

historical_beta = calculate_rolling_beta(dataCOMPANY, dataSP500, DURATION)

# Calculate annualized return for S&P 500 over the given duration
initial_value = dataSP500['Adj Close'].iloc[0]
final_value = dataSP500['Adj Close'].iloc[-1]
Rm = ((final_value / initial_value) ** (1/DURATION) - 1)
risk_free_rate = RFR/100
ERP = Rm - risk_free_rate
print(f"Equity Risk Premium: {ERP*100:.2f}%")

# Use yfinance to get the market capitalization
stock = yf.Ticker(TICKER)
market_cap = stock.info['marketCap'] / 10**9  # Convert to billions

equity_value = market_cap
print(f"The equity value (market cap) of {TICKER} is approximately ${market_cap:.2f} billion.")

# Use yfinance to get the debt values
long_term_debt = stock.balance_sheet.loc["Long Term Debt"][0] if "Long Term Debt" in stock.balance_sheet.index else 0
short_term_debt = stock.balance_sheet.loc["Short Term Debt"][0] if "Short Term Debt" in stock.balance_sheet.index else 0

# Calculate total debt
debt_value = (long_term_debt + short_term_debt) / 10**9  # Convert to billions
print(f"The total debt of {TICKER} is approximately ${debt_value:.2f} billion.")

# Use yfinance to get the cash and non-operating asset values
cash_and_cash_equivalents = stock.balance_sheet.loc["Cash And Cash Equivalents"][0] if "Cash And Cash Equivalents" in stock.balance_sheet.index else 0
# Convert to billions
cash_and_non_operating_asset = cash_and_cash_equivalents / 10**9
print(f"Cash and non-operating assets of {TICKER} is approximately ${cash_and_non_operating_asset:.2f} billion.")


df_result = get_stocks_from_same_industry(TICKER)
comparable_tickers = df_result['symbol'].tolist()

print(comparable_tickers) 
# Get unlevered betas for each comparable
unlevered_betas = [get_unlevered_beta(ticker) for ticker in comparable_tickers]
unlevered_betas = [beta for beta in unlevered_betas if beta is not None]  # Remove None values
# Calculate the industry average unlevered beta
industry_average_unlevered_beta = sum(unlevered_betas) / len(unlevered_betas)

# Estimate the terminal_unlevered_beta
omega = 0.5  # Weight given to the company's current unlevered beta
unlevered_beta = get_unlevered_beta(TICKER)
terminal_unlevered_beta = omega * unlevered_beta + (1 - omega) * industry_average_unlevered_beta

print(f"The estimated unlevered beta is: {unlevered_beta:.4f}")
print(f"The estimated terminal unlevered beta is: {terminal_unlevered_beta:.4f}")

# Linear regression model
X = np.array(range(len(historical_beta))).reshape(-1, 1)
y = historical_beta.values
model = LinearRegression().fit(X, y)
slope = model.coef_
intercept = model.intercept_

# Calculate the intersection point with terminal beta using the equation of the line
# y = mx + c; terminal_beta = slope*x + intercept
intersection_point = (terminal_unlevered_beta - intercept) / slope

# Convert intersection_point to years (assuming your historical data is daily)
intersection_in_years = intersection_point[0]/365

print(f"Expected year to converge to terminal beta: {intersection_in_years:.2f} years")

year_beta_begins_to_converge_to_terminal_beta = intersection_in_years

# Calculate the effective tax rate
income_statement = stock.financials
pretax_income = float(income_statement.loc["Pretax Income"].iloc[0])
income_tax_expense = float(income_statement.loc["Tax Provision"].iloc[0])
tax_rate = income_tax_expense / pretax_income

print(f"Current Effective Tax Rate: {tax_rate:.2%}")

current_effective_tax_rate = tax_rate

current_pretax_cost_of_debt = get_pretax_cost_of_debt(TICKER)

print(f"Current Pretax Cost of Debt: {current_pretax_cost_of_debt:.2%}")

# Get pre-tax cost of debt for each comparable
pretax_costs_of_debt = [get_pretax_cost_of_debt(ticker) for ticker in comparable_tickers]
pretax_costs_of_debt = [cost for cost in pretax_costs_of_debt if cost is not None]

# Calculate the industry average pre-tax cost of debt
industry_average_pretax_cost_of_debt = sum(pretax_costs_of_debt) / len(pretax_costs_of_debt)

# Estimate the terminal_pre_tax_cost_of_debt
omega = 0.5  # Weight given to the company's current pre-tax cost of debt
terminal_pretax_cost_of_debt = omega * current_pretax_cost_of_debt + (1 - omega) * industry_average_pretax_cost_of_debt

print(f"The estimated terminal pre-tax cost of debt is: {terminal_pretax_cost_of_debt:.2%}")

year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt = get_year_cost_of_debt_converges(TICKER, comparable_tickers)

print(f"Expected year to converge to the cost of debt: {year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt} years")

marginal_tax_rate = get_marginal_tax_rate(TICKER)

print(f"Current Marginal Tax Rate: {marginal_tax_rate:.2%}")

year_effective_tax_rate_begin_to_converge_marginal_tax_rate = 1

revenue_base = get_ttm_total_revenue(TICKER)

print(f"The total revenue of {TICKER} is approximately ${revenue_base}")

growth1, growth5 = fetch_growth_estimate(TICKER)

revenue_growth_rate_cycle1_begin = growth1/100

print(revenue_growth_rate_cycle1_begin)

revenue_growth_rate_cycle1_end = growth5/100

print(revenue_growth_rate_cycle1_end)

length_of_cylcle1 = estimate_cycle_length(revenue_growth_rate_cycle1_begin, revenue_growth_rate_cycle1_end)

print(length_of_cylcle1)

revenue_growth_rate_cycle2_begin = (revenue_growth_rate_cycle1_begin + ERP)/2

print(revenue_growth_rate_cycle2_begin)

revenue_growth_rate_cycle2_end = (revenue_growth_rate_cycle1_end + ERP)/2

print(revenue_growth_rate_cycle2_end)

length_of_cylcle2 = estimate_cycle_length(revenue_growth_rate_cycle2_begin, revenue_growth_rate_cycle2_end)

revenue_growth_rate_cycle3_begin = (revenue_growth_rate_cycle2_begin + ERP)/2

revenue_growth_rate_cycle3_end = (revenue_growth_rate_cycle2_end + ERP)/2

length_of_cylcle3 = estimate_cycle_length(revenue_growth_rate_cycle3_begin, revenue_growth_rate_cycle3_end)

revenue_convergance_periods_cycle1 = 1
revenue_convergance_periods_cycle2 = 1
revenue_convergance_periods_cycle3 = 1

current_sales_to_capital_ratio = get_sales_to_capital_ratio(TICKER)

print(current_sales_to_capital_ratio)

terminal_sales_to_capital_ratio = estimate_terminal_ratio_from_comparables(TICKER,comparable_tickers)

print(terminal_sales_to_capital_ratio)

year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital = years_to_converge(current_sales_to_capital_ratio, terminal_sales_to_capital_ratio ,0.05)

print(year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital)

current_operating_margin = get_current_operating_margin(TICKER)

print(current_operating_margin)

terminal_operating_margin = estimate_terminal_operating_margin(TICKER)

print(terminal_operating_margin)

year_operating_margin_begins_to_converge_to_terminal_operating_margin = year_margin_begins_to_converge(current_operating_margin, terminal_operating_margin, 0.02)

print(year_operating_margin_begins_to_converge_to_terminal_operating_margin)

additional_return_on_cost_of_capital_in_perpetuity = 0.02

asset_liquidation_during_negative_growth = 0

current_invested_capital = get_invested_capital(TICKER)

print(current_invested_capital)

base_case_valuation = valuator_multi_phase(
            risk_free_rate,
            ERP,
            equity_value,
            debt_value,
            cash_and_non_operating_asset,
            unlevered_beta,
            terminal_unlevered_beta,
            year_beta_begins_to_converge_to_terminal_beta,
            current_pretax_cost_of_debt,
            terminal_pretax_cost_of_debt,
            year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt,
            current_effective_tax_rate,
            marginal_tax_rate,
            year_effective_tax_rate_begin_to_converge_marginal_tax_rate,
             revenue_base,
             revenue_growth_rate_cycle1_begin,
             revenue_growth_rate_cycle1_end,
             length_of_cylcle1,
             revenue_growth_rate_cycle2_begin,
             revenue_growth_rate_cycle2_end,
             length_of_cylcle2,
             revenue_growth_rate_cycle3_begin,
             revenue_growth_rate_cycle3_end,
             length_of_cylcle3,
            revenue_convergance_periods_cycle1,
            revenue_convergance_periods_cycle2,
            revenue_convergance_periods_cycle3,
            current_sales_to_capital_ratio,
            terminal_sales_to_capital_ratio,
            year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital,
            current_operating_margin,
            terminal_operating_margin,
            year_operating_margin_begins_to_converge_to_terminal_operating_margin,
            additional_return_on_cost_of_capital_in_perpetuity,
            asset_liquidation_during_negative_growth,
            current_invested_capital)
point_estimate_describer(base_case_valuation)


df_valuation = monte_carlo_valuator_multi_phase(
    risk_free_rate = ot.Normal(0.04,.002),
    ERP =  ot.Normal(0.048,.001) ,
    equity_value = ot.Triangular(45,51.016,57),
    debt_value = ot.Triangular(3.7,3.887,4),
    unlevered_beta = ot.Triangular(.8,.9,1),
    terminal_unlevered_beta = ot.Triangular(.8,.9,1),
    year_beta_begins_to_converge_to_terminal_beta = ot.Uniform(1,2),
    current_pretax_cost_of_debt = ot.Triangular(.057,.06,.063),
    terminal_pretax_cost_of_debt = ot.Triangular(.052,.055,.058),
    year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt = ot.Uniform(1,2),
    current_effective_tax_rate = ot.Triangular(.23,.24,.25),
    marginal_tax_rate = ot.Triangular(.23,.25,.27),
    year_effective_tax_rate_begin_to_converge_marginal_tax_rate = ot.Uniform(1,3),
     revenue_base = ot.Triangular(8.8,9.2,9.6),
     revenue_growth_rate_cycle1_begin = ot.Distribution(ot.SciPyDistribution(scipy.stats.skewnorm(-2.9, loc= .145, scale=.032))), ## high growth is priced in already, skewed towards left if there is any surprise, it's most likely bad
     revenue_growth_rate_cycle1_end = ot.Distribution(ot.SciPyDistribution(scipy.stats.skewnorm(-2.9, loc=.18, scale=.033))), ## high growth, skewed towards left if there is any surprise, it's most likely bad
     length_of_cylcle1 = ot.Uniform(4,8),
     revenue_growth_rate_cycle2_begin = ot.Distribution(ot.SciPyDistribution(scipy.stats.skewnorm(-2.9, loc= .165, scale=.034))), ## high growth, skewed towards left if there is any surprise, it's most likely bad
     revenue_growth_rate_cycle2_end = ot.Distribution(ot.SciPyDistribution(scipy.stats.skewnorm(-2.9, loc= .11, scale=.032))), ## high growth, skewed towards left if there is any surprise, it's most likely bad
     length_of_cylcle2 = ot.Uniform(4,8),
     revenue_growth_rate_cycle3_begin = ot.Distribution(ot.SciPyDistribution(scipy.stats.skewnorm(-2.9, loc=.09, scale=.024))), ## high growth, skewed towards left if there is any surprise, it's most likely bad
     revenue_growth_rate_cycle3_end = ot.Normal(0.04,.002),
     length_of_cylcle3 = ot.Uniform(4,8),
    revenue_convergance_periods_cycle1 = ot.Uniform(1,2),
    revenue_convergance_periods_cycle2 = ot.Uniform(1,2),
    revenue_convergance_periods_cycle3 = ot.Uniform(1,2),
    current_sales_to_capital_ratio = ot.Triangular(1.5,1.7,1.9),
    terminal_sales_to_capital_ratio = ot.Triangular(1.1,1.3,1.6),
    year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital = ot.Uniform(1,3),
    current_operating_margin = ot.Triangular(.145,.15,.155),
    terminal_operating_margin = ot.Triangular(.12,.175,.22),
    year_operating_margin_begins_to_converge_to_terminal_operating_margin = ot.Uniform(1,3),
    additional_return_on_cost_of_capital_in_perpetuity = ot.Triangular(0.0,0.02,0.035),
    cash_and_non_operating_asset = ot.Uniform(1.6,1.8),
    asset_liquidation_during_negative_growth = ot.Uniform(0,0.000000001),
    current_invested_capital = ot.Uniform(5.8,6.2),
    sample_size= 20000,
    list_of_correlation_between_variables=[#### Intuitive / common sense  correlations
                                           ['revenue_growth_rate_cycle3_end','risk_free_rate',.95],
                                           ['revenue_growth_rate_cycle3_end','terminal_pretax_cost_of_debt',.9],
                                           ['terminal_pretax_cost_of_debt','risk_free_rate',.9],
                                           ### valuation specific correlation
                                           ['ERP','revenue_growth_rate_cycle2_begin',.4],
                                           ['ERP','revenue_growth_rate_cycle2_end',.4],
                                           ['ERP','revenue_growth_rate_cycle3_begin',.4],
                                           ['additional_return_on_cost_of_capital_in_perpetuity','terminal_sales_to_capital_ratio',0.25],
                                           ['additional_return_on_cost_of_capital_in_perpetuity','terminal_operating_margin',.5],
                                           ['terminal_sales_to_capital_ratio','terminal_operating_margin',.3],
                                           ]
                                           )


df_valuation.tail()

valuation_describer(df_valuation,
                    sharesOutstanding=.027589800)



