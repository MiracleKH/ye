import os
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.colors import LinearSegmentedColormap
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.optimizers.schedules import CosineDecay


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


np.random.seed(42)
tf.random.set_seed(42)



def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # 启用确定性操作



set_seeds()


#数据增强
class DataAugmenter:
    def __init__(self, fs=100):
        """
        初始化数据增强器
        fs: 采样频率(Hz)
        """
        self.fs = fs

    def add_gaussian_noise(self, signals, noise_level=0.05, random_state=None):

        if random_state is not None:
            np.random.seed(random_state)

        augmented = []
        for sig in signals:
            noise = np.random.normal(0, noise_level * np.std(sig), sig.shape)
            augmented.append(sig + noise)
        return np.array(augmented)

    def time_shift(self, signals, shift_range_sec=1.0, random_state=None):
        """时间偏移（左/右移动信号）"""
        if random_state is not None:
            np.random.seed(random_state)

        augmented = []
        max_shift = int(shift_range_sec * self.fs)

        for sig in signals:
            shift = np.random.randint(-max_shift, max_shift)
            if shift > 0:  # 右移（后面补零）
                augmented_sig = np.concatenate([np.zeros(shift), sig[:-shift]])
            elif shift < 0:  # 左移（前面补零）
                augmented_sig = np.concatenate([sig[-shift:], np.zeros(-shift)])
            else:
                augmented_sig = sig
            augmented.append(augmented_sig)

        return np.array(augmented)

    def amplitude_scaling(self, signals, scale_range=(0.8, 1.2), random_state=None):
        """幅度缩放"""
        if random_state is not None:
            np.random.seed(random_state)

        augmented = []
        for sig in signals:
            scale = np.random.uniform(scale_range[0], scale_range[1])
            augmented.append(sig * scale)
        return np.array(augmented)

    def random_perturbation(self, signals, perturbation_level=0.03, segment_length=50, random_state=None):
        """随机扰动（在小片段上添加随机噪声）"""
        if random_state is not None:
            np.random.seed(random_state)

        augmented = []
        for sig in signals:
            augmented_sig = sig.copy()
            num_segments = len(sig) // segment_length

            for i in range(num_segments):
                start_idx = i * segment_length
                end_idx = min((i + 1) * segment_length, len(sig))

                if np.random.random() > 0.5:  # 50%概率扰动
                    perturbation = np.random.normal(0, perturbation_level * np.std(sig),
                                                    end_idx - start_idx)
                    augmented_sig[start_idx:end_idx] += perturbation

            augmented.append(augmented_sig)
        return np.array(augmented)

    def apply_augmentation(self, signals, augmentations=None, random_state=None):

        if augmentations is None:
            augmentations = ['noise', 'shift', 'scale']

        augmented = signals.copy()

        for aug in augmentations:
            if aug == 'noise':
                augmented = self.add_gaussian_noise(augmented, random_state=random_state)
            elif aug == 'shift':
                augmented = self.time_shift(augmented, random_state=random_state)
            elif aug == 'scale':
                augmented = self.amplitude_scaling(augmented, random_state=random_state)
            elif aug == 'perturb':
                augmented = self.random_perturbation(augmented, random_state=random_state)

        return augmented


#数据预处理
class DataLoader:
    def __init__(self, data_root='./Data', augmenter=None, augmentation_factor=2):
        self.data_root = data_root
        self.labels = sorted(os.listdir(data_root))
        self.channels = ['Abdomen', 'Mask1', 'Mask2', 'Chest']
        self.augmenter = augmenter
        self.augmentation_factor = augmentation_factor

    def load_single_file(self, file_path, channel):

        if channel == 'Mask2':
            df = pd.read_excel(file_path, header=0)
            return df.iloc[:, 1].values.astype(np.float32)
        else:
            df = pd.read_csv(file_path, skiprows=8, header=None)
            return df.iloc[:, 0].values.astype(np.float32)

    def preprocess_signal(self, signal, channel):
        """信号预处理：去噪、归一化"""
        signal = self.median_filter(signal)
        signal = signal.reshape(-1, 1)
        scaler = StandardScaler() if channel == 'Mask2' else MinMaxScaler()
        return scaler.fit_transform(signal).flatten()

    def median_filter(self, signal, window_size=5):
        """中值滤波去噪"""
        pad_size = window_size // 2
        padded = np.pad(signal, pad_size, mode='edge')
        return np.array([np.median(padded[i:i + window_size]) for i in range(len(signal))])

    def load_all_data(self):
        """加载所有标签和通道的数据，合并通道并应用数据增强"""
        data = {label: [] for label in self.labels}  #按标签储存样本
        print("正在加载数据...")

        for label in tqdm(self.labels, desc="标签"):
            for ch in self.channels:
                files = sorted(os.listdir(os.path.join(self.data_root, label, ch)))[:30]
                for file in files:
                    sig = self.load_single_file(os.path.join(self.data_root, label, ch, file), ch)
                    processed_sig = self.preprocess_signal(sig, ch)


                    data[label].append(processed_sig)

                    # 数据增强
                    if self.augmenter:
                        for i in range(self.augmentation_factor):
                            augmented_sig = self.augmenter.apply_augmentation(
                                [processed_sig],
                                random_state=i  # 确保每次生成不同的增强样本
                            )[0]
                            data[label].append(augmented_sig)

        return data



class DataAnalyzer:
    @staticmethod
    def check_label_balance(data_dict):
        """检查标签平衡情况"""
        counts = {lbl: len(data_dict[lbl]) for lbl in data_dict}
        print("\n=== 标签样本数统计 ===")
        for lbl, cnt in counts.items():
            print(f"{lbl}: {cnt} 个样本")
        print(f"总样本数: {sum(counts.values())}")

    @staticmethod
    def visualize_feature_spread(features_dict, class_names, n_features=4):
        """可视化前n个特征的分布对比"""
        plt.figure(figsize=(16, 12))
        for i in range(n_features):
            plt.subplot(2, 2, i + 1)
            for lbl in class_names:
                feats = features_dict[lbl]
                plt.hist(feats[:, i], bins=30, alpha=0.6, label=lbl)
            plt.title(f"特征 {i + 1} 分布对比")
            plt.xlabel("特征值")
            plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_signal_examples(data_dict, channels, n_examples=3):
        """可视化不同类别和通道的信号示例"""
        plt.figure(figsize=(15, 10))
        for i, lbl in enumerate(data_dict.keys()):
            for j, ch in enumerate(channels):
                plt.subplot(len(data_dict), len(channels), i * len(channels) + j + 1)

                # 从每个类别中随机选择n_examples个样本
                samples = np.random.choice(len(data_dict[lbl]), n_examples, replace=False)
                for k, idx in enumerate(samples):
                    plt.plot(data_dict[lbl][idx], alpha=0.7, label=f"样本{k + 1}")

                plt.title(f"{lbl} - 随机样本")
                plt.xlabel("时间点")
                plt.ylabel("幅值")
                if k == n_examples - 1:
                    plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def calculate_feature_correlation(features_dict, channel):
        """计算特征间的相关性矩阵"""
        all_features = []
        for lbl in features_dict:
            all_features.append(features_dict[lbl])
        all_features = np.vstack(all_features)

        # 计算相关系数矩阵
        corr_matrix = np.corrcoef(all_features, rowvar=False)

        # 可视化
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', square=True)
        plt.title(f"特征相关性矩阵")
        plt.show()

        return corr_matrix


# 特征工程
class FeatureExtractor:
    def __init__(self, target_len=500, fs=100):

        self.target_len = target_len
        self.fs = fs

    def interpolate_signals(self, signals):
        """将信号插值到统一长度"""
        return np.array([np.interp(np.linspace(0, 1, self.target_len),
                                   np.linspace(0, 1, len(sig)), sig)
                         for sig in signals])

    def extract_time_domain_features(self, signals):
        """提取时域特征"""
        features = []
        for sig in signals:
            mean = np.mean(sig)
            std = np.std(sig)
            min_val = np.min(sig)
            max_val = np.max(sig)
            med = np.median(sig)
            skew = stats.skew(sig)
            kurt = stats.kurtosis(sig)
            p25 = np.percentile(sig, 25)
            p75 = np.percentile(sig, 75)
            rms = np.sqrt(np.mean(sig ** 2))
            crest = max_val / rms if rms != 0 else 0
            iqr = p75 - p25  # 四分位距
            energy = np.sum(sig ** 2)  # 信号能量
            features.append([mean, std, min_val, max_val, med, skew, kurt,
                             p25, p75, rms, crest, iqr, energy])
        return np.array(features)

    def extract_frequency_domain_features(self, signals):
        """提取频域特征"""
        features = []
        for sig in signals:
            # 傅里叶变换
            fft = np.abs(np.fft.rfft(sig))
            freq = np.fft.rfftfreq(self.target_len, 1 / self.fs)

            # 归一化功率谱
            power_spectrum = fft ** 2 / len(fft)
            total_power = np.sum(power_spectrum)


            if total_power < 1e-10:
                features.append([0, 0, 0, 0, 0, 0])
                continue

            # 主导频率
            dom_freq = freq[np.argmax(power_spectrum)]

            # 频域熵
            normalized_power = power_spectrum / total_power
            normalized_power = normalized_power[normalized_power > 0]  # 避免log(0)
            freq_entropy = -np.sum(normalized_power * np.log2(normalized_power))

            # 能量分布
            low_freq = np.sum(power_spectrum[freq < 0.5]) / total_power
            mid_freq = np.sum(power_spectrum[(freq >= 0.5) & (freq < 2)]) / total_power
            high_freq = np.sum(power_spectrum[freq >= 2]) / total_power

            features.append([dom_freq, freq_entropy, total_power, low_freq, mid_freq, high_freq])
        return np.array(features)

    def extract_derivative_features(self, signals):
        """提取导数特征"""
        d1 = np.gradient(signals, axis=1)
        d2 = np.gradient(d1, axis=1)

        # 提取一阶导数统计特征
        d1_mean = np.mean(d1, axis=1)
        d1_std = np.std(d1, axis=1)
        d1_max = np.max(d1, axis=1)
        d1_min = np.min(d1, axis=1)
        d1_abs_mean = np.mean(np.abs(d1), axis=1)

        # 提取二阶导数统计特征
        d2_mean = np.mean(d2, axis=1)
        d2_std = np.std(d2, axis=1)
        d2_max = np.max(d2, axis=1)
        d2_min = np.min(d2, axis=1)
        d2_abs_mean = np.mean(np.abs(d2), axis=1)

        return np.column_stack([d1_mean, d1_std, d1_max, d1_min, d1_abs_mean,
                                d2_mean, d2_std, d2_max, d2_min, d2_abs_mean])

    def extract_respiratory_cycle_features(self, signals):
        """提取呼吸周期特征"""
        cycle_features = []
        for sig in signals:
            # 峰值检测（寻找吸气顶点）
            peaks, _ = signal.find_peaks(sig, height=np.mean(sig) + 0.5 * np.std(sig),
                                         distance=self.fs * 0.5)  # 至少间隔0.5秒

            # 谷值检测（寻找呼气低点）
            valleys, _ = signal.find_peaks(-sig, height=-(np.mean(sig) - 0.5 * np.std(sig)),
                                           distance=self.fs * 0.5)

            if len(peaks) < 2 or len(valleys) < 2:
                # 无法检测到足够的峰谷，返回默认值
                cycle_features.append([0, 0, 0, 0, 0, 0])
                continue

            # 计算呼吸周期特征
            cycle_lengths = np.diff(peaks) / self.fs  # 周期长度(秒)
            inhale_times = []  # 吸气时间
            exhale_times = []  # 呼气时间

            for i in range(len(peaks) - 1):
                # 寻找当前峰值和下一个峰值之间的谷值
                valley_idx = np.where((valleys > peaks[i]) & (valleys < peaks[i + 1]))[0]
                if len(valley_idx) > 0:
                    nearest_valley = valleys[valley_idx[0]]
                    inhale_times.append((nearest_valley - peaks[i]) / self.fs)
                    exhale_times.append((peaks[i + 1] - nearest_valley) / self.fs)

            if not inhale_times or not exhale_times:
                cycle_features.append([0, 0, 0, 0, 0, 0])
                continue

            avg_cycle = np.mean(cycle_lengths)
            std_cycle = np.std(cycle_lengths)
            avg_inhale = np.mean(inhale_times)
            avg_exhale = np.mean(exhale_times)
            inhale_exhale_ratio = avg_inhale / (avg_exhale + 1e-10)  # 避免除以零
            cycle_rate = 60 / avg_cycle if avg_cycle > 1e-10 else 0  # 避免除以零

            cycle_features.append([avg_cycle, std_cycle, avg_inhale, avg_exhale,
                                   inhale_exhale_ratio, cycle_rate])

        return np.array(cycle_features)

    def extract_all_features(self, data_dict):
        """融合所有维度的特征"""
        features = {lbl: [] for lbl in data_dict}
        print("\n正在提取特征...")

        for lbl in tqdm(data_dict, desc="标签"):
            # 处理该标签下的所有样本
            sigs = self.interpolate_signals(data_dict[lbl])

            # 提取各类特征
            time_feats = self.extract_time_domain_features(sigs)  # 13维
            freq_feats = self.extract_frequency_domain_features(sigs)  # 6维
            deriv_feats = self.extract_derivative_features(sigs)  # 10维
            cycle_feats = self.extract_respiratory_cycle_features(sigs)  # 6维

            # 合并特征 (总计35维)
            combined = np.concatenate([time_feats, freq_feats, deriv_feats, cycle_feats], axis=1)

            # 检查并处理NaN值
            combined = np.nan_to_num(combined, nan=0.0, posinf=1e10, neginf=-1e10)

            features[lbl] = combined

        return features


#数据模块
class DataPreparer:
    def __init__(self, channels):
        self.label2idx = {}
        self.channels = channels

    def prepare_dataset(self, features_dict, strategy='group', balance=True):
        """准备数据集"""
        self.label2idx = {lbl: i for i, lbl in enumerate(features_dict.keys())}
        X, y = self._group_strategy(features_dict) if strategy == 'group' else self._avg_strategy(features_dict)

        # 数据清洗：检查并移除包含NaN或无穷大的样本
        valid_mask = np.isfinite(X).all(axis=1)
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        print(f"数据清洗：移除了 {len(X) - len(X_clean)} 个无效样本")

        if balance:
            y_labels = np.argmax(y_clean, axis=1)
            X_clean, y_labels = SMOTE(random_state=42).fit_resample(X_clean, y_labels)
            y_clean = to_categorical(y_labels)

        return train_test_split(X_clean, y_clean, test_size=0.2, random_state=42, stratify=np.argmax(y_clean, axis=1))

    def _group_strategy(self, features_dict):
        """按实验分组策略"""
        X, y = [], []
        for lbl in features_dict:
            for feat in features_dict[lbl]:
                X.append(feat)
                y.append(self.label2idx[lbl])
        return np.array(X), to_categorical(np.array(y))

    def _avg_strategy(self, features_dict):
        """平均重复实验策略"""
        X, y = [], []
        for lbl in features_dict:
            # 对所有样本的特征取平均
            avg_feat = np.mean(features_dict[lbl], axis=0)
            X.append(avg_feat)
            y.append(self.label2idx[lbl])
        return np.array(X), to_categorical(np.array(y))


#模型构建
class BreathingClassifier:
    def __init__(self, input_shape, num_classes):
        self.model = self._build_cnn_network(input_shape, num_classes)

    def _build_cnn_network(self, input_shape, num_classes):

        inputs = tf.keras.Input(shape=input_shape)

        # 特征预处理层（调整为适合CNN的输入形状）
        x = layers.Reshape((-1, 1))(inputs)

        # 多层1D卷积提取特征
        x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        x = layers.Conv1D(128, 5, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        x = layers.Conv1D(256, 5, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        # 全局池化层
        x = layers.GlobalAveragePooling1D()(x)

        # 全连接分类层
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        # 编译模型
        optimizer = optimizers.Adam(learning_rate=0.0001)
        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(multi_label=True, name='auc')]
        )

        return model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=500, batch=16):

        callbacks = [
            EarlyStopping(monitor='val_auc', patience=30, restore_best_weights=True, mode='max'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7),
            ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
        ]
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch,
            callbacks=callbacks,
            shuffle=True,
        )


#可视化
class Visualizer:
    @staticmethod
    def plot_training_history(history, save_path='./results'):

        os.makedirs(save_path, exist_ok=True)
        epochs = range(1, len(history.history['loss']) + 1)


        history_df = pd.DataFrame({
            'epoch': epochs,
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'train_accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy']
        })

        # 保存到CSV
        history_df.to_csv(os.path.join(save_path, 'training_history.csv'), index=False)
        print(f"已保存训练历史数据（含epoch）到 {save_path}/training_history.csv")

        # 绘制图像（使用epoch作为横坐标）
        plt.figure(figsize=(12, 4))

        # 准确率曲线
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history.history['accuracy'], label='训练集')
        plt.plot(epochs, history.history['val_accuracy'], label='验证集')
        plt.title('模型准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率')
        plt.legend()

        # 损失曲线
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history.history['loss'], label='训练集')
        plt.plot(epochs, history.history['val_loss'], label='验证集')
        plt.title('模型损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失值')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_history.png'))
        plt.close()


    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names, save_path='./results'):
            """绘制百分比形式的混淆矩阵并保存图像"""
            os.makedirs(save_path, exist_ok=True)

            y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(y_pred, axis=1)
            cm = confusion_matrix(y_true, y_pred)

          


            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            cm_df = pd.DataFrame(cm_percent, index=class_names, columns=class_names)
            cm_df.to_csv(os.path.join(save_path, 'confusion_matrix_raw.csv'))
            print(f"已保存原始混淆矩阵数据到 {save_path}/confusion_matrix_raw.csv")

            # 矩阵的显示设置
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.weight'] = 'bold'

            plt.figure(figsize=(15, 12))
            colors = ['#fcfeff', '#E3FAFC', '#C5F6FA', '#99E9F2', '#66D9E8', '#3BC9DB', '#22B8CF', '#15AABF', '#1098AD',
                      '#0C8599', '#0B7285']
            cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)


            sns.heatmap(cm_percent, annot=True, fmt='.1%', cmap=cmap,
                        xticklabels=class_names, yticklabels=class_names,
                        vmin=0, vmax=1, annot_kws={"size": 22})

            plt.title('Confusion Matrix', fontsize=24, fontweight='bold')
            plt.ylabel('True label', fontsize=20, fontweight='bold')
            plt.xlabel('Predicted label', fontsize=20, fontweight='bold')


            plt.xticks(fontsize=16, fontweight='bold')
            plt.yticks(fontsize=16, fontweight='bold')

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'confusion_matrix_percentage.png'), dpi=300)
            plt.close()

            # 保存分类报告
            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(os.path.join(save_path, 'classification_report.csv'))

            print("\n分类报告:")
            print(classification_report(y_true, y_pred, target_names=class_names))
    @staticmethod
    def plot_augmentation_examples(original, augmented, channel_name, label_name, save_path='./results'):

        os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(original, 'b-', alpha=0.7, label='原始信号')
        plt.title(f"{label_name} - {channel_name} - 原始信号")
        plt.xlabel("时间点")
        plt.ylabel("幅值")
        plt.legend()

        plt.subplot(2, 1, 2)
        for i, aug in enumerate(augmented):
            plt.plot(aug, alpha=0.7, label=f'增强信号 {i + 1}')
        plt.title(f"{label_name} - {channel_name} - 增强信号")
        plt.xlabel("时间点")
        plt.ylabel("幅值")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'augmentation_examples_{label_name}.png'))
        plt.close()



def main():
    # 创建结果保存目录
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)


    augmenter = DataAugmenter(fs=100)


    loader = DataLoader(augmenter=augmenter, augmentation_factor=2)  # 每个原始样本生成2个增强样本
    data = loader.load_all_data()


    analyzer = DataAnalyzer()
    analyzer.check_label_balance(data)
    analyzer.plot_signal_examples(data, loader.channels, n_examples=2)


    if augmenter:

        sample_label = list(data.keys())[0]
        sample_idx = 0
        original_signal = data[sample_label][sample_idx]

        # 生成多个增强版本
        augmented_signals = []
        for i in range(3):  # 生成3个增强样本
            augmented = augmenter.apply_augmentation(
                [original_signal],
                random_state=i,
                augmentations=['noise', 'shift', 'scale']
            )[0]
            augmented_signals.append(augmented)

        Visualizer.plot_augmentation_examples(original_signal, augmented_signals,
                                              "混合通道", sample_label, results_dir)


    extractor = FeatureExtractor(target_len=500, fs=100)  #采样频率100HZ
    features = extractor.extract_all_features(data)


    analyzer.visualize_feature_spread(features, loader.labels, n_features=4)
    analyzer.calculate_feature_correlation(features, "all")


    preparer = DataPreparer(channels=loader.channels)
    X_train, X_test, y_train, y_test = preparer.prepare_dataset(features, balance=True)


    classifier = BreathingClassifier(
        input_shape=(X_train.shape[1],),  # 输入形状为特征维度
        num_classes=len(loader.labels)
    )

    history = classifier.train_model(X_train, y_train, X_test, y_test)


    test_loss, test_acc, test_auc = classifier.model.evaluate(X_test, y_test, verbose=0)
    print(f"\n测试集准确率: {test_acc:.4f}, AUC: {test_auc:.4f}, 损失值: {test_loss:.4f}")

    # 保存评估结果
    eval_results = {
        'test_accuracy': test_acc,
        'test_auc': test_auc,
        'test_loss': test_loss
    }
    pd.DataFrame([eval_results]).to_csv(os.path.join(results_dir, 'evaluation_results.csv'), index=False)

    y_pred = classifier.model.predict(X_test)
    Visualizer.plot_training_history(history, results_dir)
    Visualizer.plot_confusion_matrix(y_test, y_pred, class_names=loader.labels, save_path=results_dir)

    # 保存模型
    classifier.model.save(os.path.join(results_dir, 'breathing_classifier.h5'))


if __name__ == "__main__":
    main()