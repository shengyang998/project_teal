//
//  AppRootView.swift
//  ProjectTeal
//
//  Created by Soleil Yu on 2025/9/23.
//

import SwiftUI
import UIKit

struct AppRootView: View {
    var body: some View {
        AppRootViewControllerRepresentable()
    }
}

struct AppRootViewControllerRepresentable: UIViewControllerRepresentable {
    func makeUIViewController(context: Context) -> AppRootViewController {
        return AppRootViewController()
    }
    
    func updateUIViewController(_ uiViewController: AppRootViewController, context: Context) {
        // No updates needed for now
    }
}

#Preview {
    AppRootView()
}
