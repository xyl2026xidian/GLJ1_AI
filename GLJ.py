# galerkin_method_complete.py - 伽辽金加权残值法完整模块（修复版）
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import integrate
from scipy.special import legendre, eval_legendre
from scipy.linalg import solve, lstsq
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 尝试导入sympy，如果失败则使用数值方法
try:
    import sympy as sp
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    st.warning("⚠️ SymPy未安装，将使用数值方法进行计算。建议安装: pip install sympy")

# ==================== 辅助函数 ====================
def compute_galerkin_numerical(n, problem_type, basis_type):
    """使用数值方法计算伽辽金解（当sympy不可用时）"""
    x = np.linspace(0, 1, 1000)
    dx = x[1] - x[0]
    
    # 定义基函数（数值版本）
    def get_basis(i, x_val):
        if basis_type == "正弦函数":
            return np.sin((i+1) * np.pi * x_val)
        elif basis_type == "多项式":
            return x_val**(i+1) * (1 - x_val)
        else:  # 勒让德
            return legendre(i+1)(2*x_val - 1) - legendre(i+1)(-1)
    
    # 定义精确解和右端项
    if problem_type == "u'' = -sin(πx)":
        u_exact = lambda x: np.sin(np.pi * x) / np.pi**2
        f = lambda x: -np.sin(np.pi * x)
    elif problem_type == "u'' = -1":
        u_exact = lambda x: x * (1 - x) / 2
        f = lambda x: -np.ones_like(x)
    elif problem_type == "u'' = -x(1-x)":
        u_exact = lambda x: x**2 * (1-x)**2 / 12 + x * (1-x) / 12
        f = lambda x: -x * (1 - x)
    else:
        u_exact = lambda x: np.sin(np.pi * x) / np.pi**2
        f = lambda x: -np.sin(np.pi * x)
    
    # 构建伽辽金系统（数值积分）
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    for i in range(n):
        phi_i = get_basis(i, x)
        dphi_i = np.gradient(phi_i, dx)
        
        for j in range(n):
            phi_j = get_basis(j, x)
            dphi_j = np.gradient(phi_j, dx)
            A[i, j] = np.trapz(dphi_i * dphi_j, x)
        
        b[i] = np.trapz(f(x) * phi_i, x)
    
    # 求解
    c = solve(A, b)
    
    # 构造近似解函数
    def u_approx(x_vals):
        result = np.zeros_like(x_vals)
        for i in range(n):
            result += c[i] * get_basis(i, x_vals)
        return result
    
    return u_approx, u_exact, c

# ==================== 第一部分：理论知识 ====================
def theory_section():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0c3483 0%, #a2b6df 100%); 
                padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;">
        <h1 style="text-align: center; font-size: 2.8rem;">🎯 伽辽金加权残值法</h1>
        <p style="text-align: center; font-size: 1.3rem;">
            加权残值法的精髓：让近似解的误差在加权意义下最小
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 理论概述
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### 📐 核心思想
        
        **加权残值法** (Weighted Residual Method) 是一种求解微分方程的近似方法。
        它不要求近似解精确满足原方程，而是让残值在某种加权意义下最小。
        
        #### 基本步骤：
        
        1. **假设近似解**：
        $$u(x) \\approx \\tilde{u}(x) = \\sum_{i=1}^{n} c_i \\phi_i(x)$$
        
        2. **定义残值**：
        $$R(x) = L[\\tilde{u}(x)] - f(x)$$
        
        3. **加权积分**：
        $$\\int_{\\Omega} w_j(x) R(x) d\\Omega = 0, \\quad j = 1,2,\\ldots,n$$
        
        4. **求解系数**：得到关于 cᵢ 的代数方程组
        """)
        
        st.info("""
        💡 **伽辽金法的特点**：
        - 权函数 = 基函数：wⱼ = φⱼ
        - 自然满足边界条件
        - 对称正定矩阵（对自共轭问题）
        - 有限元法的基础
        """)
    
    with col2:
        st.markdown("""
        ### 🔢 各种加权残值法
        
        | 方法 | 权函数 | 特点 |
        |------|--------|------|
        | **伽辽金法** | wⱼ = φⱼ | 最常用，对称性好 |
        | **配点法** | wⱼ = δ(x-xⱼ) | 简单，精度较低 |
        | **子域法** | wⱼ = 1 on Ωⱼ | 分区加权 |
        | **最小二乘法** | wⱼ = ∂R/∂cⱼ | 最小化残值平方 |
        | **矩量法** | wⱼ = xʲ | 矩匹配 |
        """)
        
        st.markdown("""
        ### 📊 收敛性条件
        
        伽辽金法的解是**最优近似**：
        - 误差在能量范数下最小
        - 随 n 增加单调收敛
        - 误差估计：
        $$||u - u_h||_E \\leq C h^{p}$$
        """)
    
    # 交互式演示
    st.markdown("### 🧪 交互式演示：伽辽金法求解一维问题")
    
    st.markdown("""
    **问题**：求解 -u''(x) = f(x)，边界条件 u(0)=u(1)=0
    
    选择基函数和项数，观察近似解的收敛过程
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        problem_type = st.selectbox(
            "选择问题",
            ["u'' = -sin(πx)", "u'' = -1", "u'' = -x(1-x)"],
            key="theory_problem"
        )
        
        n_terms = st.slider("基函数项数 n", 1, 10, 3, key="theory_n")
        
        basis_type = st.selectbox(
            "基函数类型",
            ["正弦函数", "多项式", "勒让德多项式"],
            key="theory_basis"
        )
        
        show_exact = st.checkbox("显示精确解", value=True, key="theory_exact")
        show_error = st.checkbox("显示误差分布", value=True, key="theory_error")
    
    with col2:
        st.markdown("#### 基函数定义")
        
        if basis_type == "正弦函数":
            st.latex(r"\phi_i(x) = \sin(i\pi x)")
            st.write("满足边界条件: φᵢ(0)=φᵢ(1)=0")
        elif basis_type == "多项式":
            st.latex(r"\phi_i(x) = x^i(1-x)")
            st.write("满足边界条件: φᵢ(0)=φᵢ(1)=0")
        else:
            st.latex(r"\phi_i(x) = P_{i+1}(x) - P_{i+1}(0)")
            st.write("Pᵢ 是勒让德多项式")
    
    # 求解
    if HAS_SYMPY:
        try:
            # 使用符号计算
            x_sym = sp.Symbol('x')
            
            # 定义基函数
            if basis_type == "正弦函数":
                phi = [sp.sin((i+1)*sp.pi*x_sym) for i in range(n_terms)]
            elif basis_type == "多项式":
                phi = [x_sym**(i+1)*(1-x_sym) for i in range(n_terms)]
            else:  # 勒让德
                phi = []
                for i in range(n_terms):
                    P = sp.legendre(i+1, x_sym)
                    phi.append(P - P.subs(x_sym, 0))
            
            # 定义精确解和f
            if problem_type == "u'' = -sin(πx)":
                u_exact = sp.sin(sp.pi*x_sym)/sp.pi**2
                f = -sp.sin(sp.pi*x_sym)
            elif problem_type == "u'' = -1":
                u_exact = x_sym*(1-x_sym)/2
                f = -1
            else:  # u'' = -x(1-x)
                u_exact = x_sym**2*(1-x_sym)**2/12 + x_sym*(1-x_sym)/12
                f = -x_sym*(1-x_sym)
            
            # 构造伽辽金方程组
            A = np.zeros((n_terms, n_terms))
            b = np.zeros(n_terms)
            
            for i in range(n_terms):
                for j in range(n_terms):
                    integrand = sp.diff(phi[i], x_sym) * sp.diff(phi[j], x_sym)
                    A[i, j] = float(sp.integrate(integrand, (x_sym, 0, 1)))
                
                integrand = f * phi[i]
                b[i] = float(sp.integrate(integrand, (x_sym, 0, 1)))
            
            # 求解系数
            c = solve(A, b)
            
            # 构造近似解函数
            u_approx = sum(c[i] * phi[i] for i in range(n_terms))
            u_approx_func = sp.lambdify(x_sym, u_approx, 'numpy')
            u_exact_func = sp.lambdify(x_sym, u_exact, 'numpy')
            
            # 绘图
            x_plot = np.linspace(0, 1, 200)
            u_approx_vals = u_approx_func(x_plot)
            u_exact_vals = u_exact_func(x_plot)
            
            fig, axes = plt.subplots(1, 2 if show_error else 1, figsize=(14, 5))
            
            if not show_error:
                axes = [axes]
            
            # 解的比较
            ax1 = axes[0]
            ax1.plot(x_plot, u_exact_vals, 'k--', linewidth=2, label='精确解', alpha=0.7)
            ax1.plot(x_plot, u_approx_vals, 'r-', linewidth=2, label=f'伽辽金近似 (n={n_terms})')
            
            if n_terms <= 5:
                for i in range(n_terms):
                    phi_func = sp.lambdify(x_sym, phi[i], 'numpy')
                    ax1.plot(x_plot, c[i] * phi_func(x_plot), '--', 
                            alpha=0.3, label=f'c_{i+1}φ_{i+1}')
            
            ax1.set_xlabel('x')
            ax1.set_ylabel('u(x)')
            ax1.set_title('解的比较')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 误差分布
            if show_error:
                ax2 = axes[1]
                error = np.abs(u_approx_vals - u_exact_vals)
                ax2.semilogy(x_plot, error, 'b-', linewidth=2)
                ax2.set_xlabel('x')
                ax2.set_ylabel('误差')
                ax2.set_title('误差分布')
                ax2.grid(True, alpha=0.3)
                
                max_error = np.max(error)
                mean_error = np.mean(error)
                st.info(f"📊 最大误差: {max_error:.2e}, 平均误差: {mean_error:.2e}")
            
            st.pyplot(fig)
            
            # 显示系数
            st.write("**系数 cᵢ**")
            df_coeffs = pd.DataFrame({
                'i': range(1, n_terms+1),
                'cᵢ': c
            })
            st.dataframe(df_coeffs)
            
        except Exception as e:
            st.error(f"符号计算失败: {e}")
            st.info("使用数值方法计算...")
            
            # 使用数值方法
            u_approx, u_exact, c = compute_galerkin_numerical(n_terms, problem_type, basis_type)
            
            x_plot = np.linspace(0, 1, 200)
            u_approx_vals = u_approx(x_plot)
            u_exact_vals = u_exact(x_plot)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(x_plot, u_exact_vals, 'k--', linewidth=2, label='精确解', alpha=0.7)
            ax.plot(x_plot, u_approx_vals, 'r-', linewidth=2, label=f'伽辽金近似 (n={n_terms})')
            ax.set_xlabel('x')
            ax.set_ylabel('u(x)')
            ax.set_title('解的比较 (数值方法)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.write("**系数 cᵢ**")
            df_coeffs = pd.DataFrame({
                'i': range(1, n_terms+1),
                'cᵢ': c
            })
            st.dataframe(df_coeffs)
    
    else:
        # 使用数值方法
        u_approx, u_exact, c = compute_galerkin_numerical(n_terms, problem_type, basis_type)
        
        x_plot = np.linspace(0, 1, 200)
        u_approx_vals = u_approx(x_plot)
        u_exact_vals = u_exact(x_plot)
        
        fig, axes = plt.subplots(1, 2 if show_error else 1, figsize=(14, 5))
        
        if not show_error:
            axes = [axes]
        
        ax1 = axes[0]
        ax1.plot(x_plot, u_exact_vals, 'k--', linewidth=2, label='精确解', alpha=0.7)
        ax1.plot(x_plot, u_approx_vals, 'r-', linewidth=2, label=f'伽辽金近似 (n={n_terms})')
        ax1.set_xlabel('x')
        ax1.set_ylabel('u(x)')
        ax1.set_title('解的比较 (数值方法)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if show_error:
            ax2 = axes[1]
            error = np.abs(u_approx_vals - u_exact_vals)
            ax2.semilogy(x_plot, error, 'b-', linewidth=2)
            ax2.set_xlabel('x')
            ax2.set_ylabel('误差')
            ax2.set_title('误差分布')
            ax2.grid(True, alpha=0.3)
            
            max_error = np.max(error)
            mean_error = np.mean(error)
            st.info(f"📊 最大误差: {max_error:.2e}, 平均误差: {mean_error:.2e}")
        
        st.pyplot(fig)
        
        st.write("**系数 cᵢ**")
        df_coeffs = pd.DataFrame({
            'i': range(1, n_terms+1),
            'cᵢ': c
        })
        st.dataframe(df_coeffs)

# ==================== 第二部分：多种加权残值法对比 ====================
def comparison_section():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 2rem;">
        <h2 style="text-align: center;">🔄 加权残值法对比研究</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 问题描述
    
    求解常微分方程：
    
    $$\\frac{d^2u}{dx^2} + u + x = 0, \\quad 0 < x < 1$$
    
    边界条件：u(0) = 0, u(1) = 0
    
    精确解：$$u(x) = \\frac{\\sin x}{\\sin 1} - x$$
    """)
    
    # 参数设置
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_terms = st.slider("基函数项数 n", 1, 8, 3, key="compare_n")
        method = st.selectbox(
            "加权方法",
            ["伽辽金法", "配点法", "子域法", "最小二乘法"],
            key="compare_method"
        )
    
    with col2:
        basis_type = st.selectbox(
            "基函数类型",
            ["多项式", "正弦"],
            key="compare_basis"
        )
        plot_type = st.selectbox(
            "绘图类型",
            ["解的比较", "误差分布", "收敛性研究"],
            key="compare_plot"
        )
    
    with col3:
        show_individual = st.checkbox("显示基函数分量", value=True, key="compare_show")
        show_residual = st.checkbox("显示残值分布", value=True, key="compare_residual")
    
    # 使用数值方法计算
    x = np.linspace(0, 1, 1000)
    dx = x[1] - x[0]
    
    # 定义基函数
    def get_basis(i, x_val):
        if basis_type == "多项式":
            return x_val**(i+1) * (1 - x_val)
        else:  # 正弦
            return np.sin((i+1) * np.pi * x_val)
    
    # 精确解
    u_exact = lambda x: np.sin(x) / np.sin(1) - x
    
    # 构建系统
    A = np.zeros((n_terms, n_terms))
    b = np.zeros(n_terms)
    
    if method == "伽辽金法":
        for i in range(n_terms):
            phi_i = get_basis(i, x)
            dphi_i = np.gradient(phi_i, dx)
            for j in range(n_terms):
                phi_j = get_basis(j, x)
                dphi_j = np.gradient(phi_j, dx)
                A[i, j] = np.trapz(dphi_i * dphi_j + phi_i * phi_j, x)
            b[i] = np.trapz(-x * phi_i, x)
    
    elif method == "配点法":
        points = np.linspace(0.1, 0.9, n_terms)
        for i, x_i in enumerate(points):
            for j in range(n_terms):
                phi_j = get_basis(j, x)
                d2phi_j = np.gradient(np.gradient(phi_j, dx), dx)
                # 在配点处求值
                idx = np.argmin(np.abs(x - x_i))
                A[i, j] = d2phi_j[idx] + phi_j[idx]
            b[i] = -x_i
    
    elif method == "子域法":
        subdomains = np.linspace(0, 1, n_terms+1)
        for i in range(n_terms):
            a = subdomains[i]
            b_sub = subdomains[i+1]
            mask = (x >= a) & (x <= b_sub)
            for j in range(n_terms):
                phi_j = get_basis(j, x)
                d2phi_j = np.gradient(np.gradient(phi_j, dx), dx)
                A[i, j] = np.trapz((d2phi_j + phi_j)[mask], x[mask])
            b[i] = np.trapz((-x)[mask], x[mask])
    
    else:  # 最小二乘法
        for i in range(n_terms):
            phi_i = get_basis(i, x)
            d2phi_i = np.gradient(np.gradient(phi_i, dx), dx)
            L_phi_i = d2phi_i + phi_i
            for j in range(n_terms):
                phi_j = get_basis(j, x)
                d2phi_j = np.gradient(np.gradient(phi_j, dx), dx)
                L_phi_j = d2phi_j + phi_j
                A[i, j] = np.trapz(L_phi_i * L_phi_j, x)
            b[i] = -np.trapz(L_phi_i * (-x), x)
    
    try:
        c = solve(A, b)
        
        # 构造近似解
        def u_approx(x_vals):
            result = np.zeros_like(x_vals)
            for i in range(n_terms):
                result += c[i] * get_basis(i, x_vals)
            return result
        
        x_plot = np.linspace(0, 1, 200)
        u_approx_vals = u_approx(x_plot)
        u_exact_vals = u_exact(x_plot)
        
        if plot_type == "解的比较":
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(x_plot, u_exact_vals, 'k--', linewidth=2, label='精确解', alpha=0.7)
            ax.plot(x_plot, u_approx_vals, 'r-', linewidth=2, label=f'{method} (n={n_terms})')
            
            if show_individual and n_terms <= 5:
                for i in range(n_terms):
                    ax.plot(x_plot, c[i] * get_basis(i, x_plot), '--', 
                           alpha=0.3, label=f'c_{i+1}φ_{i+1}')
            
            ax.set_xlabel('x')
            ax.set_ylabel('u(x)')
            ax.set_title(f'{method} - 解的比较')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            error = np.abs(u_approx_vals - u_exact_vals)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("最大误差", f"{np.max(error):.2e}")
            with col2:
                st.metric("平均误差", f"{np.mean(error):.2e}")
            with col3:
                st.metric("L2误差", f"{np.sqrt(np.trapz(error**2, x_plot)):.2e}")
        
        elif plot_type == "误差分布":
            fig, ax = plt.subplots(figsize=(10, 6))
            
            error = np.abs(u_approx_vals - u_exact_vals)
            ax.semilogy(x_plot, error, 'b-', linewidth=2)
            ax.set_xlabel('x')
            ax.set_ylabel('误差')
            ax.set_title(f'{method} - 误差分布 (n={n_terms})')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        else:  # 收敛性研究
            st.markdown("#### 📈 收敛性研究")
            
            n_range = range(1, min(n_terms+5, 11))
            errors = []
            
            for n_test in n_range:
                # 重新构建系统
                A_test = np.zeros((n_test, n_test))
                b_test = np.zeros(n_test)
                
                for i in range(n_test):
                    phi_i = get_basis(i, x)
                    dphi_i = np.gradient(phi_i, dx)
                    for j in range(n_test):
                        phi_j = get_basis(j, x)
                        dphi_j = np.gradient(phi_j, dx)
                        A_test[i, j] = np.trapz(dphi_i * dphi_j + phi_i * phi_j, x)
                    b_test[i] = np.trapz(-x * phi_i, x)
                
                c_test = solve(A_test, b_test)
                
                def u_test(x_vals):
                    result = np.zeros_like(x_vals)
                    for i in range(n_test):
                        result += c_test[i] * get_basis(i, x_vals)
                    return result
                
                u_test_vals = u_test(x_plot)
                error = np.max(np.abs(u_test_vals - u_exact_vals))
                errors.append(error)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.semilogy(list(n_range), errors, 'bo-', linewidth=2, markersize=8)
            ax.set_xlabel('基函数项数 n')
            ax.set_ylabel('最大误差')
            ax.set_title(f'{method} - 收敛性研究')
            ax.grid(True, alpha=0.3)
            
            if len(errors) > 2:
                log_n = np.log(list(n_range))
                log_err = np.log(errors)
                slope = np.polyfit(log_n, log_err, 1)[0]
                st.info(f"📈 收敛阶估计: {slope:.2f}")
            
            st.pyplot(fig)
        
        # 显示残值
        if show_residual:
            st.markdown("#### 🔍 残值分布")
            
            def residual(x_vals):
                u_vals = u_approx(x_vals)
                d2u = np.gradient(np.gradient(u_vals, x_vals[1]-x_vals[0]), x_vals[1]-x_vals[0])
                return d2u + u_vals + x_vals
            
            x_res = np.linspace(0.01, 0.99, 100)
            res_vals = residual(x_res)
            
            fig_res, ax_res = plt.subplots(figsize=(10, 4))
            ax_res.plot(x_res, res_vals, 'r-', linewidth=2)
            ax_res.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax_res.set_xlabel('x')
            ax_res.set_ylabel('残值 R(x)')
            ax_res.set_title(f'{method} - 残值分布 (n={n_terms})')
            ax_res.grid(True, alpha=0.3)
            st.pyplot(fig_res)
            
            res_norm = np.sqrt(np.trapz(res_vals**2, x_res))
            st.metric("残值L2范数", f"{res_norm:.2e}")
        
    except Exception as e:
        st.error(f"求解失败: {e}")
        st.info("💡 提示：尝试增加基函数项数或调整参数")

# ==================== 第三部分：工程案例 ====================
def engineering_case():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                padding: 1.5rem; border-radius: 15px; color: #333; margin-bottom: 2rem;">
        <h2 style="text-align: center;">🏭 工程案例：结构力学分析</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 问题描述：简支梁弯曲
        
    考虑一根简支梁，受均布载荷作用，其挠度满足四阶微分方程：
    
    $$EI \\frac{d^4w}{dx^4} = q(x), \\quad 0 < x < L$$
    
    边界条件（简支）：
    - w(0) = 0, w(L) = 0 (位移为零)
    - M(0) = 0, M(L) = 0 (弯矩为零)
    """)
    
    # 参数设置
    col1, col2 = st.columns(2)
    
    with col1:
        L = st.slider("梁长度 L (m)", 1.0, 10.0, 5.0)
        E = st.slider("弹性模量 E (GPa)", 10, 300, 200)
        I = st.slider("惯性矩 I (×10⁻⁶ m⁴)", 0.1, 10.0, 2.0)
        q0 = st.slider("均布载荷 q₀ (kN/m)", 1, 50, 20)
    
    with col2:
        n_terms = st.slider("基函数项数 n", 1, 10, 4, key="eng_n")
        load_type = st.selectbox(
            "载荷类型",
            ["均布载荷", "集中力"],
            key="eng_load"
        )
        show_3d = st.checkbox("显示3D变形", value=True, key="eng_3d")
    
    # 无量纲化
    E_real = E * 1e9
    I_real = I * 1e-6
    EI = E_real * I_real
    q0_real = q0 * 1000
    
    # 基函数（满足边界条件）
    x_vals = np.linspace(0, L, 500)
    dx = x_vals[1] - x_vals[0]
    
    def get_beam_basis(i, x):
        return np.sin((i+1) * np.pi * x / L)
    
    # 精确解（均布载荷）
    def exact_solution_udl(x):
        return q0_real * x**2 * (L - x)**2 / (24 * EI)
    
    # 构建伽辽金系统
    A = np.zeros((n_terms, n_terms))
    b = np.zeros(n_terms)
    
    if load_type == "均布载荷":
        q_func = lambda x: q0_real * np.ones_like(x)
        
        for i in range(n_terms):
            phi_i = get_beam_basis(i, x_vals)
            d4phi_i = np.gradient(np.gradient(np.gradient(np.gradient(phi_i, dx), dx), dx), dx)
            
            for j in range(n_terms):
                phi_j = get_beam_basis(j, x_vals)
                A[i, j] = np.trapz(EI * d4phi_i * phi_j, x_vals)
            
            b[i] = np.trapz(q_func(x_vals) * phi_i, x_vals)
    
    else:  # 集中力
        P = st.slider("集中力 P (kN)", 1, 100, 50, key="eng_P")
        P_real = P * 1000
        q_func = lambda x: P_real * (np.abs(x - L/2) < 0.01)
        
        for i in range(n_terms):
            phi_i = get_beam_basis(i, x_vals)
            d4phi_i = np.gradient(np.gradient(np.gradient(np.gradient(phi_i, dx), dx), dx), dx)
            
            for j in range(n_terms):
                phi_j = get_beam_basis(j, x_vals)
                A[i, j] = np.trapz(EI * d4phi_i * phi_j, x_vals)
            
            # 数值积分处理集中力
            idx_center = np.argmin(np.abs(x_vals - L/2))
            b[i] = P_real * phi_i[idx_center]
    
    try:
        c = solve(A, b)
        
        # 构造近似解
        def w_approx(x):
            result = np.zeros_like(x)
            for i in range(n_terms):
                result += c[i] * get_beam_basis(i, x)
            return result
        
        x_plot = np.linspace(0, L, 200)
        w_approx_vals = w_approx(x_plot)
        
        # 绘图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 挠度曲线
        ax1 = axes[0, 0]
        ax1.plot(x_plot, w_approx_vals * 1000, 'b-', linewidth=2, label='伽辽金解')
        
        if load_type == "均布载荷":
            w_exact_vals = exact_solution_udl(x_plot)
            ax1.plot(x_plot, w_exact_vals * 1000, 'r--', linewidth=2, label='精确解', alpha=0.7)
            error = np.max(np.abs(w_approx_vals - w_exact_vals))
            st.metric("最大挠度误差", f"{error*1000:.3f} mm")
        
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('挠度 w (mm)')
        ax1.set_title('梁的挠度曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 弯矩分布
        ax2 = axes[0, 1]
        # M = EI * w''
        dw2 = np.gradient(np.gradient(w_approx_vals, x_plot[1]-x_plot[0]), x_plot[1]-x_plot[0])
        M_vals = EI * dw2
        ax2.plot(x_plot, M_vals / 1000, 'r-', linewidth=2)
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('弯矩 M (kN·m)')
        ax2.set_title('弯矩分布')
        ax2.grid(True, alpha=0.3)
        
        # 应力分布
        ax3 = axes[1, 0]
        y_max = 0.2  # 假设截面高度
        sigma_vals = M_vals * y_max / I_real / 1e6
        ax3.plot(x_plot, sigma_vals, 'g-', linewidth=2)
        ax3.set_xlabel('x (m)')
        ax3.set_ylabel('最大应力 σ (MPa)')
        ax3.set_title('应力分布')
        ax3.grid(True, alpha=0.3)
        
        # 应力云图
        ax4 = axes[1, 1]
        y_section = np.linspace(-y_max, y_max, 30)
        Y, X_sec = np.meshgrid(y_section, x_plot)
        sigma_dist = np.zeros_like(X_sec)
        for i, x_i in enumerate(x_plot):
            idx = np.argmin(np.abs(x_vals - x_i))
            M_i = M_vals[i]
            sigma_dist[i, :] = M_i * Y[i, :] / I_real / 1e6
        
        contour = ax4.contourf(X_sec, Y, sigma_dist, levels=20, cmap='RdBu_r')
        plt.colorbar(contour, ax=ax4, label='应力 (MPa)')
        ax4.set_xlabel('x (m)')
        ax4.set_ylabel('截面高度 (m)')
        ax4.set_title('截面应力云图')
        ax4.set_aspect('equal')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 3D变形显示
        if show_3d:
            st.markdown("### 🎬 3D变形可视化")
            
            x_3d = np.linspace(0, L, 30)
            w_3d = w_approx(x_3d)
            scale = st.slider("变形放大倍数", 1, 200, 50, key="eng_scale")
            z_deformed = w_3d * scale
            
            fig3d = go.Figure()
            
            fig3d.add_trace(go.Scatter3d(
                x=x_3d, y=np.zeros_like(x_3d), z=z_deformed,
                mode='lines+markers',
                line=dict(color='red', width=4),
                marker=dict(size=3),
                name='变形后'
            ))
            
            fig3d.add_trace(go.Scatter3d(
                x=x_3d, y=np.zeros_like(x_3d), z=np.zeros_like(x_3d),
                mode='lines',
                line=dict(color='gray', width=2, dash='dash'),
                name='原始'
            ))
            
            fig3d.update_layout(
                scene=dict(
                    xaxis_title='x (m)',
                    yaxis_title='y (m)',
                    zaxis_title='挠度 (mm)',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
                ),
                height=500,
                title='梁的3D变形'
            )
            st.plotly_chart(fig3d, use_container_width=True)
        
        # 显示系数
        st.write("**伽辽金法系数**")
        df_coeffs = pd.DataFrame({
            'i': range(1, n_terms+1),
            'cᵢ': c
        })
        st.dataframe(df_coeffs)
        
        st.success("✅ 伽辽金法求解完成！")
        
    except Exception as e:
        st.error(f"求解失败: {e}")
        st.info("💡 提示：尝试增加基函数项数或调整参数")

# ==================== 主函数 ====================
def main():
    st.set_page_config(
        page_title="伽辽金加权残值法 - 完整教程",
        page_icon="🎯",
        layout="wide"
    )
    
    # 自定义CSS
    st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 0 20px;
        }
        .stButton button {
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # 侧边栏导航
    st.sidebar.title("📚 导航")
    st.sidebar.markdown("---")
    
    section = st.sidebar.radio(
        "选择内容",
        ["📖 理论知识", "🔄 方法对比", "🏭 工程案例"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    ### 💡 学习路径
    
    1. **理论知识** - 理解核心概念
    2. **方法对比** - 不同加权方法
    3. **工程案例** - 实际应用
    
    ### 🎯 关键概念
    - 基函数选择
    - 权函数定义
    - 残值最小化
    - 收敛性分析
    """)
    
    # 显示内容
    if section == "📖 理论知识":
        theory_section()
    elif section == "🔄 方法对比":
        comparison_section()
    else:
        engineering_case()

if __name__ == "__main__":
    main()